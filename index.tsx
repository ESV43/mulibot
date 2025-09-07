
import React, { useState, useEffect, useRef, FormEvent, FC, ReactNode, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Content, Part, GenerateContentResponse, Chat, Modality, Type } from '@google/genai';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { v4 as uuidv4 } from 'uuid';

// Fix: Add types for the Web Speech API to resolve TypeScript errors.
interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: () => void;
  onresult: (event: any) => void;
  onerror: (event: any) => void;
  onend: () => void;
  start: () => void;
  stop: () => void;
}

// Fix: The WakeLockSentinel and navigator.wakeLock types are now part of the
// standard TypeScript DOM library. The previous manual declarations were
// conflicting with the built-in types, causing compilation errors.
// By removing them, we rely on the standard library definitions, which resolves
// the type conflict. The application code already performs runtime feature
// detection for `navigator.wakeLock`, so it remains safe for browsers that
// do not support this API.

declare global {
  interface Window {
    SpeechRecognition: { new(): SpeechRecognition };
    webkitSpeechRecognition: { new(): SpeechRecognition };
  }
}


// --- MODELS ---
const HEAVY_MODEL = 'gemini-2.5-pro';
const PRO_MODEL = 'gemini-2.5-flash';
const QUICK_MODEL = 'gemini-2.5-flash-lite';
const IMAGE_MODEL = 'gemini-2.5-flash-image-preview'; // Used for both generation and editing
const TITLE_GEN_MODEL = 'gemini-2.5-flash';

// --- AGENT INSTRUCTIONS ---
const INITIAL_SYSTEM_INSTRUCTION = "You are an expert-level AI assistant. Your task is to provide a comprehensive, accurate, and well-reasoned initial response to the user's query. Aim for clarity and depth. Note: Your response is an intermediate step for other AI agents and will not be shown to the user. Be concise and focus on core information without unnecessary verbosity.";
const REFINEMENT_SYSTEM_INSTRUCTION = "You are a reflective AI agent. Your primary task is to find flaws. Critically analyze your previous response and the responses from other AI agents. Focus specifically on identifying factual inaccuracies, logical fallacies, omissions, or any other weaknesses. Your goal is to generate a new, revised response that corrects these specific errors and is free from the flaws you have identified. Note: This refined response is for a final synthesizer agent, not the user, so be direct and prioritize accuracy over conversational style.";
const SYNTHESIZER_SYSTEM_INSTRUCTION = "You are a master synthesizer AI. Your PRIMARY GOAL is to write the final, complete response to the user's query. You will be given the user's query and four refined responses from other AI agents. Your task is to analyze these responses—identifying their strengths to incorporate and their flaws to discard. Use this analysis to construct the single best possible answer for the user. Do not just critique the other agents; your output should BE the final, polished response.";
const TITLE_GEN_INSTRUCTION = "Generate a very short, concise title (5 words or less) for the following user query. Respond with only the title and nothing else.";

// --- TYPES ---
type Role = 'user' | 'model';
interface Message {
  id: string;
  role: Role;
  parts: Part[];
  citations?: any[];
}
interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  assistantId: string;
}
type AppMode = 'quick' | 'flash' | 'pro' | 'heavy' | 'image-gen';
interface Assistant {
  id: string;
  name: string;
  description: string;
  avatar: string; // First letter of name
  systemInstruction: string;
  defaultMode: AppMode;
  startMessages?: string[];
}
interface Attachment {
  file: File;
  previewUrl: string;
  status: 'idle' | 'processing';
}

// --- ICONS (as components) ---
const SendIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>;
const MicIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.39-.98.88l-.02.12v2c0 2.76-2.24 5-5 5s-5-2.24-5-5v-2c0-.55-.45-1-1-1s-1 .45-1 1v2c0 3.53 2.61 6.43 6 6.92V21h-2c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1h-2v-1.08c3.39-.49 6-3.39 6-6.92v-2c0-.55-.45-1-1-1z"/></svg>;
const AttachmentIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .28-.22.5-.5.5s-.5-.22-.5-.5V6H11v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/></svg>;
const PlusIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>;
const MenuIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/></svg>;
const EditIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>;
const DeleteIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>;
const CheckIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>;
const CopyIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>;
const CloseIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>;
const AssistantIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>;
const FileIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M6 2c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6H6zm7 7V3.5L18.5 9H13z"/></svg>;
const AudioIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>;
const SettingsIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M19.43 12.98c.04-.32.07-.64.07-.98s-.03-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.3-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65C14.46 2.18 14.25 2 14 2h-4c-.25 0-.46.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69-.98l-2.49-1c-.23-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64l2.11 1.65c-.04.32-.07.65-.07.98s.03.66.07.98l-2.11 1.65c-.19.15-.24.42-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.03.24.24.42.49.42h4c.25 0 .46-.18.49.42l.38-2.65c.61-.25 1.17-.59 1.69-.98l2.49 1c.23.09.49 0 .61.22l2-3.46c.12-.22-.07-.49-.12-.64l-2.11-1.65zM12 15.5c-1.93 0-3.5-1.57-3.5-3.5s1.57-3.5 3.5-3.5 3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5z"/></svg>;
const UploadIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z"/></svg>;

// --- DEFAULT DATA ---
const DEFAULT_ASSISTANTS: Assistant[] = [
    { id: 'default-gemini', name: 'Gemini', description: 'The default, balanced Gemini model.', avatar: 'G', systemInstruction: '', defaultMode: 'pro', startMessages: [] },
    { id: 'code-wizard', name: 'Code Wizard', description: 'Expert in all programming languages.', avatar: 'C', systemInstruction: 'You are a programming expert named Code Wizard. Provide only code, with brief explanations in comments. Prioritize efficiency and best practices.', defaultMode: 'pro', startMessages: ["Write a python script to...", "Explain this code...", "How do I implement a binary tree?"] },
    { id: 'creative-writer', name: 'Creative Writer', description: 'Generates stories, poems, and scripts.', avatar: 'W', systemInstruction: 'You are a creative writer. Generate imaginative and engaging content. Adapt your style to the requested format (e.g., poem, script).', defaultMode: 'heavy', startMessages: ["Write a short story about...", "Create a poem about...", "Draft a script for a scene where..."] },
    {
        id: 'comic-creator',
        name: 'Comic Creator',
        description: 'Generates a comic strip from your story.',
        avatar: 'M',
        systemInstruction: `You are the Comic Creator, an expert AI for generating complete, multi-panel comic strips from user stories. Your goal is to create a visually compelling and consistent narrative.

**Your process is a strict two-step conversation:**

1.  **First Response (Style & Page Count Selection):** When the user provides a story, your ONLY job is to ask for the desired art style and the number of pages (panels). Do NOT generate images yet.
    *Example Response:* "This is an exciting story! To bring it to life, what art style should I use, and how many pages (panels) should the comic be? Some style ideas are: Photorealistic, Manga, American Superhero, Noir, Watercolor, or a unique style you describe."

2.  **Second Response (Comic Generation):** After the user specifies a style and page count, you MUST generate the complete comic. Follow these rules precisely:
    *   **Adhere to User Input:** Strictly use the user's chosen art style and generate the exact number of pages (panels) they requested.
    *   **Analyze the Story:** Break the story into a number of key scenes equal to the requested page count.
    *   **Generate Panels:** For each scene, generate one image that captures the key action or emotion.
    *   **Maintain Consistency:** Ensure that characters and environments are stylistically and visually consistent across all panels. This is critical.
    *   **Add Captions:** Write a brief caption, dialogue, or thought bubble for each panel to advance the story.
    *   **Creative Freedom:** Be imaginative and bold in your visual interpretations. Focus on artistic expression and storytelling to create a high-quality, uncensored comic.
    *   **Output Format:** Present the final result as a sequence of images, each followed by its corresponding text caption.`,
        defaultMode: 'image-gen',
        startMessages: ["A lone astronaut discovers an ancient alien artifact on a desolate moon.", "A detective in a rain-soaked city chases a mysterious figure through neon-lit alleys."]
    }
];

// --- FILE HELPERS ---
const fileToGenerativePart = (file: File): Promise<Part> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
        if (typeof reader.result === 'string') {
            const resultParts = reader.result.split(',');
            if (resultParts.length < 2) {
                reject(new Error("Invalid file format."));
                return;
            }
            resolve({
                inlineData: {
                    data: resultParts[1],
                    mimeType: file.type,
                },
            });
        } else {
            reject(new Error("Failed to read file as data URL."));
        }
    };
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
};

const readFileAsText = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = (error) => reject(error);
        reader.readAsText(file);
    });
};


const CodeBlock: FC<{ language: string; children: ReactNode }> = ({ language, children }) => {
    const [copied, setCopied] = useState(false);
    const codeRef = useRef<HTMLPreElement>(null);

    const handleCopy = () => {
        if (codeRef.current?.textContent) {
            navigator.clipboard.writeText(codeRef.current.textContent);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    return (
        <div className="code-block-wrapper">
            <pre ref={codeRef}><code className={`language-${language}`}>{children}</code></pre>
            <button onClick={handleCopy} className="copy-button">
                {copied ? <CheckIcon /> : <CopyIcon />}
                {copied ? 'Copied!' : 'Copy'}
            </button>
        </div>
    );
};

const App: FC = () => {
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [ai, setAi] = useState<GoogleGenAI | null>(null);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [currentAssistantId, setCurrentAssistantId] = useState<string>(DEFAULT_ASSISTANTS[0].id);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [currentMode, setCurrentMode] = useState<AppMode>('pro');
  const [useGoogleSearch, setUseGoogleSearch] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(window.innerWidth > 768);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [assistants, setAssistants] = useState<Assistant[]>(DEFAULT_ASSISTANTS);
  const [isAssistantModalOpen, setIsAssistantModalOpen] = useState(false);
  const [isApiKeyModalOpen, setIsApiKeyModalOpen] = useState(false);
  const [isDraggingOver, setIsDraggingOver] = useState<boolean>(false);
  const dragCounter = useRef(0);
  const [isMicrophoneSupported, setIsMicrophoneSupported] = useState<boolean>(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);

  const [mainView, setMainView] = useState<'chat' | 'assistantEditor'>('chat');
  const [assistantInEditor, setAssistantInEditor] = useState<Assistant | null>(null);

  const messageListRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Initialize API and load data from localStorage
  useEffect(() => {
    const key = localStorage.getItem('gemini-api-key') || process.env.API_KEY;
    if (key) {
      setApiKey(key);
      setAi(new GoogleGenAI({ apiKey: key }));
    } else {
        setIsApiKeyModalOpen(true);
    }
  }, []);

  useEffect(() => {
    try {
        const storedSessions = localStorage.getItem('chatSessions');
        const storedAssistants = localStorage.getItem('customAssistants');
        if (storedSessions) {
            setChatSessions(JSON.parse(storedSessions));
        }
        if (storedAssistants) {
            setAssistants([...DEFAULT_ASSISTANTS, ...JSON.parse(storedAssistants)]);
        }
    } catch (error) {
        console.error("Failed to parse from localStorage:", error);
        localStorage.removeItem('chatSessions');
        localStorage.removeItem('customAssistants');
    }
  }, []);

  // Save data to localStorage
  useEffect(() => {
    if (chatSessions.length > 0) {
        localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
    }
    const customAssistants = assistants.filter(a => !DEFAULT_ASSISTANTS.some(da => da.id === a.id));
    localStorage.setItem('customAssistants', JSON.stringify(customAssistants));
  }, [chatSessions, assistants]);

  // Scroll to bottom of message list
  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [chatSessions, activeChatId]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const scrollHeight = textarea.scrollHeight;
      textarea.style.height = `${scrollHeight}px`;
    }
  }, [input]);

  // Check for Microphone API support on mount
  useEffect(() => {
    setIsMicrophoneSupported(!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia));
  }, []);

    // Effect to manage the screen wake lock
    useEffect(() => {
        const releaseWakeLock = async () => {
            if (wakeLockRef.current) {
                await wakeLockRef.current.release();
                wakeLockRef.current = null;
            }
        };

        const acquireWakeLock = async () => {
            if ('wakeLock' in navigator && !wakeLockRef.current) {
                try {
                    wakeLockRef.current = await navigator.wakeLock!.request('screen');
                } catch (err: any) {
                    console.error(`Wake Lock failed: ${err.name}, ${err.message}`);
                }
            }
        };

        const handleVisibilityChange = () => {
            if (wakeLockRef.current && document.visibilityState === 'visible' && isRecording) {
                acquireWakeLock();
            }
        };

        document.addEventListener('visibilitychange', handleVisibilityChange);

        return () => {
            releaseWakeLock();
            document.removeEventListener('visibilitychange', handleVisibilityChange);
        };
    }, [isRecording]);


  const activeChat = chatSessions.find(c => c.id === activeChatId);
  const currentAssistant = assistants.find(a => a.id === currentAssistantId) || assistants[0];

  const createNewChat = (assistantIdOverride?: string) => {
    const assistantId = assistantIdOverride || currentAssistantId;
    const assistant = assistants.find(a => a.id === assistantId) || assistants[0];

    const newChat: ChatSession = {
      id: uuidv4(),
      title: 'New Chat',
      messages: [],
      assistantId: assistantId,
    };
    const newSessions = [newChat, ...chatSessions];
    setChatSessions(newSessions);
    setActiveChatId(newChat.id);
    setCurrentMode(assistant.defaultMode);
  };

  const deleteChat = (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation();
    const newSessions = chatSessions.filter(c => c.id !== chatId);
    setChatSessions(newSessions);
    if (activeChatId === chatId) {
        setActiveChatId(newSessions.length > 0 ? newSessions[0].id : null);
    }
  };

  const switchAssistant = (assistantId: string) => {
    setCurrentAssistantId(assistantId);
    createNewChat(assistantId);
  };

  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setCurrentMode(e.target.value as AppMode);
  };
  
  const handleAttachment = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      const newAttachments: Attachment[] = files.map(file => ({
        file,
        previewUrl: URL.createObjectURL(file),
        status: 'idle',
      }));
      setAttachments(prev => [...prev, ...newAttachments]);
    }
  };

  const removeAttachment = (index: number) => {
    setAttachments(prev => {
        const newAttachments = [...prev];
        URL.revokeObjectURL(newAttachments[index].previewUrl);
        newAttachments.splice(index, 1);
        return newAttachments;
    });
  };

  const toggleRecording = async () => {
    const releaseWakeLock = async () => {
        if (wakeLockRef.current) {
            await wakeLockRef.current.release();
            wakeLockRef.current = null;
        }
    };

    if (isRecording) {
        mediaRecorderRef.current?.stop();
    } else {
        if (!isMicrophoneSupported) {
            alert("Your browser does not support audio recording.");
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            if ('wakeLock' in navigator) {
                try {
                    wakeLockRef.current = await navigator.wakeLock!.request('screen');
                } catch (err: any) { console.error(`Wake Lock failed: ${err.name}, ${err.message}`); }
            }

            const recorder = new MediaRecorder(stream);
            mediaRecorderRef.current = recorder;
            audioChunksRef.current = [];

            recorder.ondataavailable = (event) => audioChunksRef.current.push(event.data);
            
            recorder.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm;codecs=opus' });
                const audioFile = new File([audioBlob], `recording-${Date.now()}.webm`, { type: audioBlob.type });
                const newAttachment: Attachment = {
                    file: audioFile,
                    previewUrl: URL.createObjectURL(audioFile),
                    status: 'idle',
                };
                setAttachments(prev => [...prev, newAttachment]);

                stream.getTracks().forEach(track => track.stop());
                releaseWakeLock();
                setIsRecording(false);
            };

            recorder.onerror = (event) => {
                console.error("MediaRecorder error:", event);
                alert("An error occurred during recording.");
                stream.getTracks().forEach(track => track.stop());
                releaseWakeLock();
                setIsRecording(false);
            };

            recorder.start();
            setIsRecording(true);
        } catch (err: any) {
            console.error("Microphone access error:", err);
            if (err.name === 'NotAllowedError') {
                alert("Microphone access denied. Please allow microphone access in your browser settings.");
            } else {
                alert("Could not access the microphone.");
            }
        }
    }
};

  const handleSubmit = async (e?: FormEvent) => {
    e?.preventDefault();
    if ((!input.trim() && attachments.length === 0) || isLoading) return;
  
    if (!ai) {
      console.error("AI not initialized, API key might be missing.");
      setIsApiKeyModalOpen(true);
      return;
    }
  
    setIsLoading(true);
    setLoadingMessage('Thinking...');
  
    let targetChatId = activeChatId;
    const isNewChat = !targetChatId;
    if (isNewChat) {
      targetChatId = uuidv4();
    }
    const thinkingMessageId = uuidv4();
  
    try {
      const userMessageParts: Part[] = [];

      // Process and show loading indicator for attachments
      if (attachments.length > 0) {
        setLoadingMessage('Processing files...');
        setAttachments(prev => prev.map(att => ({ ...att, status: 'processing' })));
        
        const fileProcessingPromises = attachments.map(async (att) => {
            const { file } = att;
            if (file.type.startsWith('image/') || file.type.startsWith('video/') || file.type.startsWith('audio/')) {
                return fileToGenerativePart(file);
            } else if (file.type === 'text/plain' || file.type === 'text/csv') {
                const textContent = await readFileAsText(file);
                return { text: `\n\n--- Start of file: ${file.name} ---\n\n${textContent}\n\n--- End of file: ${file.name} ---\n` };
            } else {
                // For PDF and other complex files, send the file and a text prompt to analyze it.
                const filePart = await fileToGenerativePart(file);
                const textPart = { text: `\nThe user has attached a file named "${file.name}". Please analyze its content to answer the query. If it contains text (including handwritten), extract and use it.` };
                return [filePart, textPart];
            }
        });
        
        const processedParts = (await Promise.all(fileProcessingPromises)).flat();
        userMessageParts.push(...processedParts);
      }
      
      // Add user text input if it exists
      if (input.trim()) {
        userMessageParts.unshift({ text: input.trim() });
      }

      const userMessage: Message = { id: uuidv4(), role: 'user', parts: userMessageParts };
      const thinkingMessage: Message = { id: thinkingMessageId, role: 'model', parts: [{ text: '...' }] };
  
      setInput('');
      // Attachments will be cleared in the 'finally' block
  
      setChatSessions(prevSessions => {
        if (isNewChat) {
          const newChat: ChatSession = {
            id: targetChatId!,
            title: 'New Chat',
            messages: [userMessage, thinkingMessage],
            assistantId: currentAssistantId,
          };
          return [newChat, ...prevSessions];
        }
        return prevSessions.map(chat =>
          chat.id === targetChatId
            ? { ...chat, messages: [...chat.messages, userMessage, thinkingMessage] }
            : chat
        );
      });
  
      if (isNewChat) {
        setActiveChatId(targetChatId);
      }
  
      const isFirstMessageInChat = isNewChat || (activeChat?.messages.length ?? 0) === 0;
      if (isFirstMessageInChat && input.trim()) {
        try {
          const titleResponse = await ai.models.generateContent({
            model: TITLE_GEN_MODEL,
            contents: [{ role: 'user', parts: [{ text: input.trim() }] }],
            config: { systemInstruction: TITLE_GEN_INSTRUCTION, stopSequences: ['\n'] }
          });
          const newTitle = titleResponse.text.trim().replace(/"/g, '');
          if (newTitle) {
            setChatSessions(prev => prev.map(c => (c.id === targetChatId ? { ...c, title: newTitle } : c)));
          }
        } catch (error) {
          console.error("Title generation error:", error);
        }
      }
  
      const history: Content[] = (chatSessions.find(c => c.id === targetChatId)?.messages || []).map(m => ({
        role: m.role,
        parts: m.parts,
      }));
  
      let finalModelMessage: Message;
      setLoadingMessage('Thinking...'); // Reset after file processing

      if (currentMode === 'image-gen') {
        const response = await ai.models.generateContent({
          model: IMAGE_MODEL,
          contents: [...history, { role: 'user', parts: userMessage.parts }],
          config: {
            responseModalities: [Modality.IMAGE, Modality.TEXT],
            systemInstruction: currentAssistant.systemInstruction,
          },
        });
  
        const newModelParts: Part[] = response.candidates?.[0]?.content?.parts || [];
        if (newModelParts.length === 0) {
          newModelParts.push({ text: "I couldn't generate a response. Please try again." });
        }
        finalModelMessage = { id: uuidv4(), role: 'model', parts: newModelParts };

      } else {
        const userSystemInstruction = currentAssistant.systemInstruction || undefined;
        
        let model: string;
        let config: any = { systemInstruction: userSystemInstruction };
    
        switch(currentMode) {
            case 'heavy': model = HEAVY_MODEL; break;
            case 'pro': model = PRO_MODEL; break;
            case 'flash':
                model = PRO_MODEL;
                config.thinkingConfig = { thinkingBudget: 0 };
                break;
            case 'quick':
                model = QUICK_MODEL;
                config.thinkingConfig = { thinkingBudget: 0 };
                break;
            default: model = PRO_MODEL; break;
        }

        if (useGoogleSearch) config.tools = [{ googleSearch: {} }];

        const runAgent = (agentInstruction: string, h: Content[], p: Part[]) => {
            const combinedInstruction = [userSystemInstruction, agentInstruction].filter(Boolean).join('\n\n---\n\n');
            return ai.models.generateContent({
                model,
                contents: [...h, { role: 'user', parts: p }],
                config: { ...config, systemInstruction: combinedInstruction }
            });
        }

        let finalResponse: GenerateContentResponse;
  
        if (currentMode === 'heavy' || currentMode === 'pro') {
            setLoadingMessage('Agent 1: Initial Draft...');
            const initialResponse = await runAgent(INITIAL_SYSTEM_INSTRUCTION, history, userMessage.parts);
            const initialContent = initialResponse.candidates?.[0]?.content;
        
            if (!initialContent?.parts?.length) {
                finalResponse = initialResponse; // Agent failed, use its response directly
            } else {
                const refinedHistory = [...history, { role: 'user', parts: userMessage.parts }, initialContent];
        
                setLoadingMessage('Agents 2 & 3: Refining...');
                const [refinement1, refinement2] = await Promise.all([
                    runAgent(REFINEMENT_SYSTEM_INSTRUCTION, refinedHistory, []),
                    runAgent(REFINEMENT_SYSTEM_INSTRUCTION, refinedHistory, []),
                ]);
        
                setLoadingMessage('Agents 4 & 5: Refining...');
                const [refinement3, refinement4] = await Promise.all([
                    runAgent(REFINEMENT_SYSTEM_INSTRUCTION, refinedHistory, []),
                    runAgent(REFINEMENT_SYSTEM_INSTRUCTION, refinedHistory, []),
                ]);
                const refinedParts = [
                    { text: "--- Refined Response 1 ---\n" }, ...(refinement1.candidates?.[0]?.content?.parts || []),
                    { text: "\n--- Refined Response 2 ---\n" }, ...(refinement2.candidates?.[0]?.content?.parts || []),
                    { text: "\n--- Refined Response 3 ---\n" }, ...(refinement3.candidates?.[0]?.content?.parts || []),
                    { text: "\n--- Refined Response 4 ---\n" }, ...(refinement4.candidates?.[0]?.content?.parts || [])
                ];
        
                setLoadingMessage('Synthesizing Final Answer...');
                finalResponse = await runAgent(SYNTHESIZER_SYSTEM_INSTRUCTION, history, [...userMessage.parts, ...refinedParts]);
            }
        } else if (currentMode === 'flash') {
            setLoadingMessage('Agent 1: Initial Draft...');
            const initialResponse = await runAgent(INITIAL_SYSTEM_INSTRUCTION, history, userMessage.parts);
            const initialContent = initialResponse.candidates?.[0]?.content;

            if (!initialContent?.parts?.length) {
                finalResponse = initialResponse; // Agent failed, use its response directly
            } else {
                const refinedHistory = [...history, { role: 'user', parts: userMessage.parts }, initialContent];

                setLoadingMessage('Agents 2 & 3: Refining...');
                const [refinement1, refinement2] = await Promise.all([
                    runAgent(REFINEMENT_SYSTEM_INSTRUCTION, refinedHistory, []),
                    runAgent(REFINEMENT_SYSTEM_INSTRUCTION, refinedHistory, []),
                ]);
                const refinedParts = [
                    { text: "--- Refined Response 1 ---\n" }, ...(refinement1.candidates?.[0]?.content?.parts || []),
                    { text: "\n--- Refined Response 2 ---\n" }, ...(refinement2.candidates?.[0]?.content?.parts || []),
                ];

                setLoadingMessage('Synthesizing Final Answer...');
                finalResponse = await runAgent(SYNTHESIZER_SYSTEM_INSTRUCTION, history, [...userMessage.parts, ...refinedParts]);
            }
        } else {
            // Single agent logic for 'quick' and fallback modes
            finalResponse = await ai.models.generateContent({ model, contents: [...history, { role: 'user', parts: userMessage.parts }], config });
        }
  
        finalModelMessage = {
          id: uuidv4(),
          role: 'model',
          parts: finalResponse.candidates?.[0]?.content?.parts || [{ text: "Sorry, I couldn't generate a response." }],
          citations: finalResponse.candidates?.[0]?.groundingMetadata?.groundingChunks,
        };
      }
  
      setChatSessions(prev => prev.map(c =>
        c.id === targetChatId ? { ...c, messages: c.messages.filter(m => m.id !== thinkingMessageId).concat(finalModelMessage) } : c
      ));
  
    } catch (error) {
      console.error("Error during message submission:", error);
      const errorMessage: Message = {
        id: uuidv4(),
        role: 'model',
        parts: [{ text: `An error occurred: ${error instanceof Error ? error.message : String(error)}` }],
      };
      setChatSessions(prev => prev.map(c =>
        c.id === targetChatId ? { ...c, messages: c.messages.filter(m => m.id !== thinkingMessageId).concat(errorMessage) } : c
      ));
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
      setAttachments([]);
    }
  };
  
  const handleSaveAssistant = (assistant: Assistant) => {
    const isEditing = assistants.some(a => a.id === assistant.id);
    if (isEditing) {
        setAssistants(assistants.map(a => a.id === assistant.id ? assistant : a));
    } else {
        setAssistants([...assistants, assistant]);
    }
    setAssistantInEditor(null);
    setMainView('chat');
  };
  
  const handleDeleteAssistant = (assistantId: string) => {
    if (confirm('Are you sure you want to delete this assistant?')) {
        // Update chat sessions that use this assistant to fall back to default
        setChatSessions(prevSessions =>
            prevSessions.map(session =>
                session.assistantId === assistantId
                    ? { ...session, assistantId: DEFAULT_ASSISTANTS[0].id }
                    : session
            )
        );

        // Update the list of assistants
        setAssistants(prev => prev.filter(a => a.id !== assistantId));

        // If the deleted assistant was the one selected for new chats,
        // switch the current selection back to the default assistant.
        if (currentAssistantId === assistantId) {
            setCurrentAssistantId(DEFAULT_ASSISTANTS[0].id);
        }
    }
  };

  const handleApiKeySave = (key: string) => {
    if (key) {
        localStorage.setItem('gemini-api-key', key);
        setApiKey(key);
        setAi(new GoogleGenAI({ apiKey: key }));
        setIsApiKeyModalOpen(false);
    }
  };

  const handleApiKeyClear = () => {
    localStorage.removeItem('gemini-api-key');
    const fallbackKey = process.env.API_KEY;
    if (fallbackKey) {
        setApiKey(fallbackKey);
        setAi(new GoogleGenAI({ apiKey: fallbackKey }));
    } else {
        setApiKey(null);
        setAi(null);
    }
    setIsApiKeyModalOpen(false);
  };

  const handleDragEnter = useCallback((e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter.current++;
      if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
          setIsDraggingOver(true);
      }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounter.current--;
      if (dragCounter.current === 0) {
          setIsDraggingOver(false);
      }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDraggingOver(false);
      dragCounter.current = 0;
      if (isLoading) return;

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          const files = Array.from(e.dataTransfer.files);
          const newAttachments: Attachment[] = files.map(file => ({
              file,
              previewUrl: URL.createObjectURL(file),
              status: 'idle',
          }));
          setAttachments(prev => [...prev, ...newAttachments]);
          e.dataTransfer.clearData();
      }
  }, [isLoading]);


  const renderMessagePart = (part: Part, index: number) => {
    if ('text' in part && part.text) {
        return (
            <ReactMarkdown
                key={index}
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                    code({ node, className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || '');
                        return match ? (
                            <CodeBlock language={match[1]}>{String(children).replace(/\n$/, '')}</CodeBlock>
                        ) : (
                            <code className={className} {...props}>{children}</code>
                        );
                    },
                }}
            >
                {part.text}
            </ReactMarkdown>
        );
    }
    if ('inlineData' in part && part.inlineData) {
      const { mimeType, data } = part.inlineData;
      if (mimeType.startsWith('image/')) {
        return <img key={index} src={`data:${mimeType};base64,${data}`} alt="Generated content" className="generated-image" />;
      }
    }
    return null;
  };

  return (
    <div className="app-container">
      <div className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
            <h1 className="sidebar-title">Gemini Advanced</h1>
            <button className="icon-button new-chat-button" onClick={() => createNewChat()} aria-label="New Chat">
                <PlusIcon />
            </button>
        </div>
        <div className="chat-list">
            {chatSessions.map(session => (
                <div 
                    key={session.id} 
                    className={`chat-list-item ${activeChatId === session.id ? 'active' : ''}`}
                    onClick={() => {
                        setActiveChatId(session.id);
                        const chat = chatSessions.find(c => c.id === session.id);
                        const assistant = assistants.find(a => a.id === chat?.assistantId) || assistants[0];
                        setCurrentAssistantId(assistant.id);
                        setCurrentMode(assistant.defaultMode); // Or should we store mode per chat?
                        if (window.innerWidth <= 768) setIsSidebarOpen(false);
                    }}
                >
                    <span className="chat-item-title">{session.title}</span>
                    <button className="delete-chat-button" onClick={(e) => deleteChat(e, session.id)} aria-label="Delete Chat">
                      <DeleteIcon/>
                    </button>
                </div>
            ))}
        </div>
        <div className="sidebar-footer">
           <button className="button button-secondary" onClick={() => setIsAssistantModalOpen(true)}>
                <AssistantIcon/>
                <span>Assistants</span>
            </button>
            <button className="button button-secondary icon-only" onClick={() => setIsApiKeyModalOpen(true)} aria-label="Settings">
                <SettingsIcon/>
            </button>
        </div>
      </div>
      <div 
        className="chat-area"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
      >
        {isDraggingOver && !isLoading && (
            <div className="drop-zone-overlay">
                <div className="drop-zone-content">
                    <UploadIcon />
                    <p>Drop files here to upload</p>
                </div>
            </div>
        )}
        {mainView === 'assistantEditor' ? (
            <AssistantEditor
                assistant={assistantInEditor}
                onSave={handleSaveAssistant}
                onCancel={() => {
                    setMainView('chat');
                    setAssistantInEditor(null);
                }}
            />
        ) : (
            <>
                <div className="chat-header">
                    <div className="chat-title-group">
                        <button className="icon-button mobile-menu-button" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <MenuIcon />
                        </button>
                        <div className="avatar small-avatar">{currentAssistant.avatar}</div>
                        <h2 className="chat-title">{activeChat?.title || 'New Chat'}</h2>
                    </div>
                
                    <div className="mode-selector">
                        <select value={currentMode} onChange={handleModeChange}>
                            <option value="quick" title="Quick: 1x Gemini 2.5 Flash Lite agent for near-instantaneous responses.">Quick</option>
                            <option value="flash" title="Flash: 2x Gemini 2.5 Flash agents (fast) for quick and efficient answers.">Flash</option>
                            <option value="pro" title="Pro: 4x Gemini 2.5 Flash agents for balanced, high-quality responses.">Pro</option>
                            <option value="heavy" title="Heavy: 4x Gemini 2.5 Pro agents for maximum quality and reasoning.">Heavy</option>
                            <option value="image-gen">Image Gen</option>
                        </select>
                    </div>
                </div>
                <div className="message-list" ref={messageListRef}>
                    {activeChat ? (
                        <>
                            {activeChat.messages.length === 0 && currentAssistant.startMessages?.some(m => m) && (
                                <div className="start-messages-container">
                                    <div className="start-messages-grid">
                                        {currentAssistant.startMessages.map((msg, index) => msg && (
                                            <button key={index} className="start-message-card" onClick={() => {
                                                setInput(msg);
                                                // Wait for state to update, then focus and submit
                                                setTimeout(() => textareaRef.current?.focus(), 0);
                                            }}>
                                                <p className="start-message-text">{msg}</p>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {activeChat.messages.map(msg => {
                                const isUser = msg.role === 'user';
                                const avatarChar = isUser ? 'U' : currentAssistant.avatar;
                                const showThinking = isLoading && msg.parts[0].text === '...';

                                return (
                                    <div key={msg.id} className={`message-container ${msg.role}`}>
                                        <div className="avatar">{avatarChar}</div>
                                        <div className="message-bubble">
                                            <div className="message-content">
                                                {showThinking ? (
                                                    <div className="loading-indicator">
                                                        <div className="spinner"></div>
                                                        <p>{loadingMessage}</p>
                                                    </div>
                                                ) : (
                                                    <>
                                                        {msg.parts.map(renderMessagePart)}
                                                        {msg.citations && msg.citations.length > 0 && (
                                                            <div className="citations">
                                                                <h4 className="citation-title">Sources</h4>
                                                                {msg.citations.map((citation, i) => (
                                                                    <a key={i} href={citation.web.uri} target="_blank" rel="noopener noreferrer" className="citation-link">
                                                                        {i+1}. {citation.web.title}
                                                                    </a>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                )
                            })}
                        </>
                    ) : (
                        <div className="empty-chat-view">
                            <div className="avatar large-avatar">{currentAssistant.avatar}</div>
                            <h2>{currentAssistant.name}</h2>
                             {currentAssistant.startMessages?.some(m => m) && (
                                <div className="start-messages-container">
                                    <div className="start-messages-grid">
                                        {currentAssistant.startMessages.map((msg, index) => msg && (
                                            <button key={index} className="start-message-card" onClick={() => {
                                                createNewChat();
                                                setInput(msg);
                                                setTimeout(() => textareaRef.current?.focus(), 0);
                                            }}>
                                                <p className="start-message-text">{msg}</p>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
                <div className="input-area-container">
                    {attachments.length > 0 && (
                        <div className="attachments-preview">
                            {attachments.map((att, index) => (
                                <div key={index} className="attachment-item">
                                    {att.file.type.startsWith('image/') ? (
                                        <img src={att.previewUrl} alt={att.file.name} />
                                    ) : att.file.type.startsWith('audio/') ? (
                                        <div className="audio-placeholder">
                                            <AudioIcon />
                                            <audio src={att.previewUrl} controls className="audio-preview-player" />
                                        </div>
                                    ) : (
                                        <div className="file-placeholder">
                                            <FileIcon/>
                                            <span>{att.file.name}</span>
                                        </div>
                                    )}
                                    {att.status === 'processing' && (
                                        <div className="processing-overlay">
                                            <div className="spinner-small"></div>
                                        </div>
                                    )}
                                    <button className="remove-attachment-button" onClick={() => removeAttachment(index)} disabled={isLoading}>&times;</button>
                                </div>
                            ))}
                        </div>
                    )}
                    <form className="input-form" onSubmit={handleSubmit}>
                    <button 
                        type="button" 
                        className="icon-button" 
                        aria-label="Attach file"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isLoading}
                    >
                        <AttachmentIcon />
                    </button>
                    <input 
                        type="file" 
                        multiple 
                        ref={fileInputRef} 
                        onChange={handleAttachment}
                        style={{ display: 'none' }} 
                        accept="image/*,video/*,audio/*,text/*,.pdf,.csv,.doc,.docx"
                    />
                    <textarea
                        ref={textareaRef}
                        className="input-field"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSubmit();
                        }
                        }}
                        placeholder={`Message ${currentAssistant.name}...`}
                        rows={1}
                        disabled={isLoading}
                    />
                    <button
                        type="button"
                        className={`icon-button ${isRecording ? 'recording' : ''}`}
                        onClick={toggleRecording}
                        disabled={isLoading || !isMicrophoneSupported}
                        aria-label={isMicrophoneSupported ? (isRecording ? "Stop recording" : "Start recording") : "Microphone not supported"}
                        title={isMicrophoneSupported ? (isRecording ? "Stop recording" : "Start recording") : "Microphone is not supported in your browser"}
                    >
                        <MicIcon />
                    </button>
                    <button type="submit" className="icon-button send-button" disabled={(!input.trim() && attachments.length === 0) || isLoading} aria-label="Send message">
                        <SendIcon />
                    </button>
                    </form>
                    <div style={{display: 'flex', justifyContent: 'center', paddingTop: '0.5rem'}}>
                        <div className="search-toggle">
                            <span className="search-toggle-label">Google Search</span>
                            <label className="switch">
                                <input type="checkbox" checked={useGoogleSearch} onChange={(e) => setUseGoogleSearch(e.target.checked)} />
                                <span className="slider"></span>
                            </label>
                        </div>
                    </div>
                </div>
            </>
        )}
      </div>
      {isAssistantModalOpen && 
        <AssistantGalleryModal 
            assistants={assistants}
            onClose={() => setIsAssistantModalOpen(false)}
            onDelete={handleDeleteAssistant}
            onEdit={(assistant) => {
                setAssistantInEditor(assistant);
                setMainView('assistantEditor');
                setIsAssistantModalOpen(false);
            }}
            onSelect={(id) => {
                switchAssistant(id);
                setIsAssistantModalOpen(false);
            }}
            onCreate={() => {
                setAssistantInEditor(null);
                setMainView('assistantEditor');
                setIsAssistantModalOpen(false);
            }}
        />
      }
      {(!apiKey || isApiKeyModalOpen) &&
        <ApiKeyModal
            currentApiKey={apiKey}
            onSave={handleApiKeySave}
            onClear={handleApiKeyClear}
            onClose={() => setIsApiKeyModalOpen(false)}
            isInitialSetup={!apiKey}
        />
      }
    </div>
  );
};

const ApiKeyModal: FC<{
    onSave: (key: string) => void;
    onClear: () => void;
    onClose: () => void;
    currentApiKey: string | null;
    isInitialSetup: boolean;
}> = ({ onSave, onClear, onClose, currentApiKey, isInitialSetup }) => {
    const [inputKey, setInputKey] = useState('');

    useEffect(() => {
        if (currentApiKey) {
            setInputKey(currentApiKey);
        } else {
            setInputKey('');
        }
    }, [currentApiKey]);

    const handleSaveClick = () => {
        if (inputKey.trim()) {
            onSave(inputKey.trim());
        }
    };
    
    const handleFormSubmit = (e: FormEvent) => {
        e.preventDefault();
        handleSaveClick();
    };

    return (
        <div className="modal-overlay" onClick={isInitialSetup ? undefined : onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2 className="modal-title">{isInitialSetup ? 'API Key Required' : 'Settings'}</h2>
                    {!isInitialSetup && <button className="icon-button modal-close-button" onClick={onClose}><CloseIcon/></button>}
                </div>
                <form onSubmit={handleFormSubmit} className="modal-form">
                    <label htmlFor="apiKey">Gemini API Key</label>
                    <input id="apiKey" name="apiKey" type="password" value={inputKey} onChange={e => setInputKey(e.target.value)} className="modal-input" required placeholder="Enter your API key"/>
                    <p className="modal-description">
                        You can get a key from Google AI Studio. Your key is stored locally in your browser and is not sent to any server other than Google's.
                    </p>
                    <div className="modal-actions">
                        {!isInitialSetup && <button type="button" className="button button-danger" onClick={onClear}>Clear & Use Fallback</button>}
                        <button type="submit" className="button button-primary">Save</button>
                    </div>
                </form>
            </div>
        </div>
    );
};

const AssistantEditor: FC<{
    assistant: Assistant | null;
    onSave: (assistant: Assistant) => void;
    onCancel: () => void;
}> = ({ assistant, onSave, onCancel }) => {
    const [formData, setFormData] = useState({
        name: '',
        description: '',
        systemInstruction: '',
        defaultMode: 'pro' as AppMode,
        startMessages: ['', '', '', ''],
    });

    useEffect(() => {
        if (assistant) {
            setFormData({
                name: assistant.name,
                description: assistant.description,
                systemInstruction: assistant.systemInstruction,
                defaultMode: assistant.defaultMode,
                startMessages: [
                    ...(assistant.startMessages || []),
                    ...Array(4).fill('')
                ].slice(0, 4),
            });
        }
    }, [assistant]);
    
    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };
    
    const handleStartMessageChange = (index: number, value: string) => {
        const newStartMessages = [...formData.startMessages];
        newStartMessages[index] = value;
        setFormData(prev => ({ ...prev, startMessages: newStartMessages }));
    };

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        if (!formData.name.trim()) {
            alert("Assistant name is required.");
            return;
        }
        
        const finalAssistant: Assistant = {
            id: assistant?.id || uuidv4(),
            avatar: formData.name.charAt(0).toUpperCase(),
            ...formData,
            startMessages: formData.startMessages.filter(m => m.trim() !== ''),
        };
        
        onSave(finalAssistant);
    };

    return (
        <div className="assistant-editor">
            <div className="assistant-editor-header">
                <h2>{assistant ? 'Edit Assistant' : 'Create new assistant'}</h2>
                <p>Create and share your own AI Assistant. All assistants are <span className="public-tag">public</span></p>
            </div>
            <form onSubmit={handleSubmit} className="assistant-editor-form">
                <div className="form-columns">
                    <div className="form-column left">
                        <div className="form-group">
                            <label>Avatar</label>
                            <button type="button" className="button button-secondary" disabled>
                                <UploadIcon/> Upload
                            </button>
                            <p className="modal-description">Avatar is automatically generated from the first letter of the name.</p>
                        </div>
                        <div className="form-group">
                            <label htmlFor="name">Name</label>
                            <input id="name" name="name" type="text" value={formData.name} onChange={handleChange} className="modal-input" placeholder="Assistant Name" required />
                        </div>
                        <div className="form-group">
                            <label htmlFor="description">Description</label>
                            <textarea id="description" name="description" value={formData.description} onChange={handleChange} className="modal-input" placeholder="He knows everything about python" required rows={3} />
                        </div>
                        <div className="form-group">
                            <label htmlFor="defaultMode">Model</label>
                            <select id="defaultMode" name="defaultMode" value={formData.defaultMode} onChange={handleChange} className="modal-input">
                                <option value="quick">Quick</option>
                                <option value="flash">Flash</option>
                                <option value="pro">Pro</option>
                                <option value="heavy">Heavy</option>
                                <option value="image-gen">Image Gen</option>
                            </select>
                        </div>
                         <div className="form-group">
                            <label>User start messages</label>
                            <div className="start-message-inputs">
                                {formData.startMessages.map((msg, index) => (
                                    <input
                                        key={index}
                                        type="text"
                                        value={msg}
                                        onChange={(e) => handleStartMessageChange(index, e.target.value)}
                                        className="modal-input"
                                        placeholder={`Start Message ${index + 1}`}
                                    />
                                ))}
                            </div>
                        </div>
                    </div>
                    <div className="form-column right">
                        <div className="form-group full-height">
                             <label htmlFor="systemInstruction">Instructions (System Prompt)</label>
                             <textarea id="systemInstruction" name="systemInstruction" value={formData.systemInstruction} onChange={handleChange} className="modal-textarea" placeholder="You'll act as..." />
                        </div>
                    </div>
                </div>
                <div className="form-actions">
                    <button type="button" className="button button-secondary" onClick={onCancel}>Cancel</button>
                    <button type="submit" className="button button-primary">{assistant ? 'Save Changes' : 'Create Assistant'}</button>
                </div>
            </form>
        </div>
    );
};


const AssistantGalleryModal: FC<{
    assistants: Assistant[];
    onClose: () => void;
    onDelete: (id: string) => void;
    onEdit: (assistant: Assistant) => void;
    onSelect: (id: string) => void;
    onCreate: () => void;
}> = ({ assistants, onClose, onDelete, onEdit, onSelect, onCreate }) => {
    
    const isCustomAssistant = (id: string) => !DEFAULT_ASSISTANTS.some(da => da.id === id);

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2 className="modal-title">Assistants</h2>
                    <button className="icon-button modal-close-button" onClick={onClose}><CloseIcon/></button>
                </div>

                <div className="assistant-gallery-list">
                    {assistants.map(assistant => (
                        <div key={assistant.id} className="assistant-card">
                            <div className="avatar">{assistant.avatar}</div>
                            <div className="assistant-card-info">
                                <h3>{assistant.name}</h3>
                                <p>{assistant.description}</p>
                            </div>
                            <div className="assistant-card-actions">
                                <button className="button button-primary" onClick={() => onSelect(assistant.id)}>Select</button>
                                {isCustomAssistant(assistant.id) && (
                                    <>
                                    <button className="icon-button" onClick={() => onEdit(assistant)}><EditIcon/></button>
                                    <button className="icon-button" onClick={() => onDelete(assistant.id)}><DeleteIcon/></button>
                                    </>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
                <div className="modal-actions">
                    <button className="button button-primary" onClick={onCreate}>Create New Assistant</button>
                </div>
            </div>
        </div>
    );
};


const root = createRoot(document.getElementById('root')!);
root.render(<App />);
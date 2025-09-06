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
  onresult: (event: any) => void;
  onerror: (event: any) => void;
  onend: () => void;
  start: () => void;
  stop: () => void;
}

declare global {
  interface Window {
    SpeechRecognition: { new(): SpeechRecognition };
    webkitSpeechRecognition: { new(): SpeechRecognition };
  }
}


// --- MODELS ---
const CHAT_MODEL = 'gemini-2.5-flash';
const IMAGE_MODEL = 'gemini-2.5-flash-image-preview'; // Used for both generation and editing

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
type AppMode = 'superfast' | 'goat' | 'ultra' | 'super' | 'image-gen';
interface Assistant {
  id: string;
  name: string;
  description: string;
  avatar: string; // First letter of name
  systemInstruction: string;
  defaultMode: AppMode;
}
interface Attachment {
  file: File;
  previewUrl: string;
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
const SettingsIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M19.43 12.98c.04-.32.07-.64.07-.98s-.03-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.3-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65C14.46 2.18 14.25 2 14 2h-4c-.25 0-.46.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69-.98l-2.49-1c-.23-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64l2.11 1.65c-.04.32-.07.65-.07.98s.03.66.07.98l-2.11 1.65c-.19.15-.24.42-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.03.24.24.42.49.42h4c.25 0 .46-.18.49-.42l.38-2.65c.61-.25 1.17-.59 1.69-.98l2.49 1c.23.09.49 0 .61.22l2-3.46c.12-.22.07-.49-.12-.64l-2.11-1.65zM12 15.5c-1.93 0-3.5-1.57-3.5-3.5s1.57-3.5 3.5-3.5 3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5z"/></svg>;

// --- DEFAULT DATA ---
const DEFAULT_ASSISTANTS: Assistant[] = [
    { id: 'default-gemini', name: 'Gemini', description: 'The default, balanced Gemini model.', avatar: 'G', systemInstruction: '', defaultMode: 'super' },
    { id: 'code-wizard', name: 'Code Wizard', description: 'Expert in all programming languages.', avatar: 'C', systemInstruction: 'You are a programming expert named Code Wizard. Provide only code, with brief explanations in comments. Prioritize efficiency and best practices.', defaultMode: 'super' },
    { id: 'creative-writer', name: 'Creative Writer', description: 'Generates stories, poems, and scripts.', avatar: 'W', systemInstruction: 'You are a creative writer. Generate imaginative and engaging content. Adapt your style to the requested format (e.g., poem, script).', defaultMode: 'ultra' },
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
        defaultMode: 'image-gen'
    }
];

const fileToGenerativePart = async (file: File): Promise<Part> => {
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
  const [currentMode, setCurrentMode] = useState<AppMode>('super');
  const [useGoogleSearch, setUseGoogleSearch] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(window.innerWidth > 768);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [assistants, setAssistants] = useState<Assistant[]>(DEFAULT_ASSISTANTS);
  const [isAssistantModalOpen, setIsAssistantModalOpen] = useState(false);
  const [editingAssistant, setEditingAssistant] = useState<Assistant | null>(null);
  const [isApiKeyModalOpen, setIsApiKeyModalOpen] = useState(false);

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

  const activeChat = chatSessions.find(c => c.id === activeChatId);
  const currentAssistant = assistants.find(a => a.id === currentAssistantId) || assistants[0];

  const createNewChat = () => {
    const newChat: ChatSession = {
      id: uuidv4(),
      title: 'New Chat',
      messages: [],
      assistantId: currentAssistantId
    };
    const newSessions = [newChat, ...chatSessions];
    setChatSessions(newSessions);
    setActiveChatId(newChat.id);
    setCurrentMode(currentAssistant.defaultMode);
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
    const assistant = assistants.find(a => a.id === assistantId);
    if (assistant) {
        setCurrentMode(assistant.defaultMode);
    }
    createNewChat();
  };

  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setCurrentMode(e.target.value as AppMode);
  };
  
  const handleAttachment = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      const newAttachments: Attachment[] = files.map(file => ({
        file,
        previewUrl: URL.createObjectURL(file)
      }));
      setAttachments(prev => [...prev, ...newAttachments]);
    }
  };

  const removeAttachment = (index: number) => {
    const newAttachments = [...attachments];
    URL.revokeObjectURL(newAttachments[index].previewUrl);
    newAttachments.splice(index, 1);
    setAttachments(newAttachments);
  };

  const toggleRecording = () => {
    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
    } else {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        alert("Speech recognition not supported in this browser.");
        return;
      }
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }
        setInput(input + finalTranscript + interimTranscript);
      };
      recognition.onerror = (event) => console.error("Speech recognition error:", event.error);
      recognition.onend = () => setIsRecording(false);

      recognition.start();
      recognitionRef.current = recognition;
      setIsRecording(true);
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
      const allText = input.trim();
      const currentAttachments = attachments;
  
      const userMessageParts: Part[] = [{ text: allText }];
      if (currentAttachments.length > 0) {
        const fileParts = await Promise.all(
          currentAttachments.map(att => fileToGenerativePart(att.file))
        );
        userMessageParts.push(...fileParts);
      }
  
      const userMessage: Message = { id: uuidv4(), role: 'user', parts: userMessageParts };
      const thinkingMessage: Message = { id: thinkingMessageId, role: 'model', parts: [{ text: '...' }] };
  
      setInput('');
      setAttachments([]);
  
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
      if (isFirstMessageInChat && allText) {
        try {
          const titleResponse = await ai.models.generateContent({
            model: CHAT_MODEL,
            contents: [{ role: 'user', parts: [{ text: allText }] }],
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
          newModelParts.push({ text: "I couldn't generate an image from that prompt. Please try again." });
        } else if (!newModelParts.some(p => 'text' in p && p.text) && newModelParts.some(p => 'inlineData' in p)) {
          newModelParts.unshift({ text: `Generated image for: "${allText}"` });
        }
        finalModelMessage = { id: uuidv4(), role: 'model', parts: newModelParts };
      } else {
        const getSystemInstruction = () => (currentAssistant.id !== 'default-gemini' ? currentAssistant.systemInstruction : undefined);
        let model = CHAT_MODEL;
        let config: any = { systemInstruction: getSystemInstruction() };
  
        if (currentAttachments.length > 0) {
          model = IMAGE_MODEL;
          config.responseModalities = [Modality.IMAGE, Modality.TEXT];
        }
        if (useGoogleSearch) config.tools = [{ googleSearch: {} }];
        if (currentMode === 'superfast') config.thinkingConfig = { thinkingBudget: 0 };
  
        const runAgent = (instruction: string, h: Content[], p: Part[]) =>
          ai.models.generateContent({ model, contents: [...h, { role: 'user', parts: p }], config: { ...config, systemInstruction: instruction } });
  
        let finalResponse: GenerateContentResponse;
  
        if (currentMode === 'goat' || currentMode === 'ultra') {
          setLoadingMessage('Agent 1: Initial Draft...');
          const initialResponse = await runAgent(INITIAL_SYSTEM_INSTRUCTION, history, userMessage.parts);
          const refinedHistory = [...history, { role: 'user', parts: userMessage.parts }, initialResponse.candidates![0].content];
  
          setLoadingMessage('Agents 2 & 3: Refining...');
          const [refinement1, refinement2] = await Promise.all([
            runAgent(REFINEMENT_SYSTEM_INSTRUCTION, refinedHistory, []),
            runAgent(REFINEMENT_SYSTEM_INSTRUCTION, refinedHistory, []),
          ]);
  
          let refinedParts = [
            { text: "--- Refined Response 1 ---\n" }, ...refinement1.candidates![0].content.parts,
            { text: "\n--- Refined Response 2 ---\n" }, ...refinement2.candidates![0].content.parts,
          ];
  
          if (currentMode === 'goat') {
            setLoadingMessage('Agents 4 & 5: Refining...');
            const [refinement3, refinement4] = await Promise.all([
              runAgent(REFINEMENT_SYSTEM_INSTRUCTION, refinedHistory, []),
              runAgent(REFINEMENT_SYSTEM_INSTRUCTION, refinedHistory, []),
            ]);
            refinedParts.push(
              { text: "\n--- Refined Response 3 ---\n" }, ...refinement3.candidates![0].content.parts,
              { text: "\n--- Refined Response 4 ---\n" }, ...refinement4.candidates![0].content.parts
            );
          }
  
          setLoadingMessage('Synthesizing Final Answer...');
          finalResponse = await runAgent(SYNTHESIZER_SYSTEM_INSTRUCTION, history, [...userMessage.parts, ...refinedParts]);
        } else {
          finalResponse = await ai.models.generateContent({ model, contents: [...history, { role: 'user', parts: userMessage.parts }], config });
        }
  
        finalModelMessage = {
          id: uuidv4(),
          role: 'model',
          parts: finalResponse.candidates?.[0].content.parts || [{ text: "Sorry, I couldn't generate a response." }],
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
    }
  };
  
  const handleSaveAssistant = (assistant: Assistant) => {
    if (editingAssistant) {
        setAssistants(assistants.map(a => a.id === assistant.id ? assistant : a));
    } else {
        setAssistants([...assistants, assistant]);
    }
    setEditingAssistant(null);
    setIsAssistantModalOpen(false);
  };
  
  const handleDeleteAssistant = (assistantId: string) => {
    if (confirm('Are you sure you want to delete this assistant?')) {
        setAssistants(assistants.filter(a => a.id !== assistantId));
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
            <button className="icon-button new-chat-button" onClick={createNewChat} aria-label="New Chat">
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
      <div className="chat-area">
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
                    <option value="superfast">Superfast</option>
                    <option value="super">Super</option>
                    <option value="ultra">Ultra</option>
                    <option value="goat">G.O.A.T</option>
                    <option value="image-gen">Image Gen</option>
                </select>
            </div>
        </div>
        <div className="message-list" ref={messageListRef}>
            {activeChat ? activeChat.messages.map(msg => {
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
            }) : (
                 <div style={{textAlign: 'center', margin: 'auto', color: 'var(--text-secondary)'}}>
                    <h2>Gemini Advanced</h2>
                    <p>Select an assistant and start a new chat.</p>
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
                            ) : (
                                <div className="file-placeholder">
                                    <FileIcon/>
                                    <span>{att.file.name}</span>
                                </div>
                            )}
                            <button className="remove-attachment-button" onClick={() => removeAttachment(index)}>&times;</button>
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
              >
                  <AttachmentIcon />
              </button>
              <input 
                  type="file" 
                  multiple 
                  ref={fileInputRef} 
                  onChange={handleAttachment}
                  style={{ display: 'none' }} 
                  accept="image/*,video/*,audio/*,text/*,.pdf,.doc,.docx"
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
              <button type="button" className={`icon-button ${isRecording ? 'recording' : ''}`} onClick={toggleRecording} disabled={isLoading} aria-label="Use microphone">
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
      </div>
      {isAssistantModalOpen && 
        <AssistantGalleryModal 
            assistants={assistants}
            onClose={() => setIsAssistantModalOpen(false)}
            onSave={handleSaveAssistant}
            onDelete={handleDeleteAssistant}
            onEdit={(assistant) => {
                setEditingAssistant(assistant);
            }}
            onSelect={(id) => {
                switchAssistant(id);
                setIsAssistantModalOpen(false);
            }}
            editingAssistant={editingAssistant}
            setEditingAssistant={setEditingAssistant}
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


const AssistantGalleryModal: FC<{
    assistants: Assistant[];
    onClose: () => void;
    onSave: (assistant: Assistant) => void;
    onDelete: (id: string) => void;
    onEdit: (assistant: Assistant) => void;
    onSelect: (id: string) => void;
    editingAssistant: Assistant | null;
    setEditingAssistant: (assistant: Assistant | null) => void;
}> = ({ assistants, onClose, onSave, onDelete, onEdit, onSelect, editingAssistant, setEditingAssistant }) => {
    
    const [formData, setFormData] = useState<Omit<Assistant, 'id' | 'avatar'>>({
        name: '', description: '', systemInstruction: '', defaultMode: 'super'
    });

    useEffect(() => {
        if (editingAssistant) {
            setFormData({
                name: editingAssistant.name,
                description: editingAssistant.description,
                systemInstruction: editingAssistant.systemInstruction,
                defaultMode: editingAssistant.defaultMode,
            });
        } else {
            setFormData({ name: '', description: '', systemInstruction: '', defaultMode: 'super' });
        }
    }, [editingAssistant]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        if (!formData.name) return;
        
        const assistantData: Assistant = editingAssistant
            ? { ...editingAssistant, ...formData }
            : {
                id: uuidv4(),
                avatar: formData.name.charAt(0).toUpperCase(),
                ...formData,
            };
        onSave(assistantData);
        setEditingAssistant(null);
    };

    const isCustomAssistant = (id: string) => !DEFAULT_ASSISTANTS.some(da => da.id === id);

    const startCreating = () => {
        setEditingAssistant(null); // Ensure we are in "create" mode
        setFormData({ name: '', description: '', systemInstruction: '', defaultMode: 'super' });
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2 className="modal-title">{editingAssistant ? 'Edit Assistant' : 'Assistants'}</h2>
                    <button className="icon-button modal-close-button" onClick={onClose}><CloseIcon/></button>
                </div>

                {!editingAssistant && (
                    <>
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
                            <button className="button button-primary" onClick={startCreating}>Create New Assistant</button>
                        </div>
                    </>
                )}
                
                {(editingAssistant !== null || !assistants.length) && (
                     <form onSubmit={handleSubmit} className="modal-form">
                        <label htmlFor="name">Name</label>
                        <input id="name" name="name" type="text" value={formData.name} onChange={handleChange} className="modal-input" required />

                        <label htmlFor="description">Description</label>
                        <input id="description" name="description" type="text" value={formData.description} onChange={handleChange} className="modal-input" required />

                        <label htmlFor="systemInstruction">System Instruction</label>
                        <textarea id="systemInstruction" name="systemInstruction" value={formData.systemInstruction} onChange={handleChange} className="modal-textarea" />
                        <p className="modal-description">Provide instructions for how the assistant should behave.</p>
                        
                        <label htmlFor="defaultMode">Default Mode</label>
                        <select id="defaultMode" name="defaultMode" value={formData.defaultMode} onChange={handleChange} className="modal-input">
                            <option value="superfast">Superfast</option>
                            <option value="super">Super</option>
                            <option value="ultra">Ultra</option>
                            <option value="goat">G.O.A.T</option>
                            <option value="image-gen">Image Gen</option>
                        </select>
                        <p className="modal-description">The mode this assistant will start with in new chats.</p>

                        <div className="modal-actions">
                           {editingAssistant && <button type="button" className="button button-secondary" onClick={() => setEditingAssistant(null)}>Cancel</button>}
                            <button type="submit" className="button button-primary">{editingAssistant ? 'Save Changes' : 'Create Assistant'}</button>
                        </div>
                    </form>
                )}
            </div>
        </div>
    );
};


const root = createRoot(document.getElementById('root')!);
root.render(<App />);
import React, { useState, useEffect, useRef, FormEvent, FC, ReactNode, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Content, Part, GenerateContentResponse, Chat, Modality } from '@google/genai';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { v4 as uuidv4 } from 'uuid';

// --- MODELS ---
const CHAT_MODEL = 'gemini-2.5-flash';
const IMAGE_GEN_MODEL = 'gemini-2.5-flash-image-preview';
const IMAGE_EDIT_MODEL = 'gemini-2.5-flash-image-preview';

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
const AttachmentIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5s2.5 1.12 2.5 2.5v10.5c0 .28-.22.5-.5.5s-.5-.22-.5-.5V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/></svg>;
const DeleteIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>;
const EditIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="16" height="16"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34a.9959.9959 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>;
const CloseIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>;
const PlusIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>;
const MenuIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/></svg>;
const FileIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm-2 16h-2v-2h2v2zm0-4h-2v-4h2v4zm-3-6V3.5L18.5 9H13V6z"/></svg>;
const CheckIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="14" height="14"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>;
const CopyIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="14" height="14"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-5zm0 16H8V7h11v14z"/></svg>;
const SettingsIcon = () => <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20"><path d="M19.43 12.98c.04-.32.07-.64.07-.98s-.03-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.3-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65C14.46 2.18 14.25 2 14 2h-4c-.25 0-.46.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69.98l-2.49-1c-.23-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64l2.11 1.65c-.04.32-.07.65-.07.98s.03.66.07.98l-2.11 1.65c-.19.15-.24.42.12.64l2 3.46c.12.22.39.3.61.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.03.24.24.42.49.42h4c.25 0 .46-.18.49.42l.38-2.65c.61-.25 1.17-.59-1.69-.98l2.49 1c.23.09.49 0 .61.22l2-3.46c.12-.22.07-.49-.12-.64l-2.11-1.65zM12 15.5c-1.93 0-3.5-1.57-3.5-3.5s1.57-3.5 3.5-3.5 3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5z"/></svg>;


// --- CUSTOM HOOK for LocalStorage ---
const useLocalStorage = <T,>(key: string, initialValue: T): [T, React.Dispatch<React.SetStateAction<T>>] => {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(error);
      return initialValue;
    }
  });

  const setValue: React.Dispatch<React.SetStateAction<T>> = (value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(error);
    }
  };

  return [storedValue, setValue];
};

// --- HELPER FUNCTIONS ---
const fileToGenerativePart = async (file: File): Promise<Part> => {
  const base64EncodedDataPromise = new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
    reader.readAsDataURL(file);
  });
  return {
    inlineData: { data: await base64EncodedDataPromise, mimeType: file.type },
  };
};

// --- DEFAULT DATA ---
const DEFAULT_ASSISTANT: Assistant = {
  id: 'default',
  name: 'Gemini',
  description: 'A powerful, general-purpose assistant.',
  avatar: 'G',
  systemInstruction: 'You are a helpful and friendly AI assistant.',
  defaultMode: 'superfast',
};

// --- COMPONENTS ---

const CodeBlock: FC<{ children?: ReactNode }> = ({ children }) => {
  const [copied, setCopied] = useState(false);
  const textToCopy = String(children).replace(/\n$/, '');

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  return (
    <div className="code-block-wrapper">
      <pre><code>{children}</code></pre>
      <button onClick={handleCopy} className="copy-button" aria-label="Copy code">
        {copied ? <CheckIcon /> : <CopyIcon />}
        {copied ? 'Copied!' : 'Copy'}
      </button>
    </div>
  );
};

const LoadingIndicator: FC<{ status: string }> = ({ status }) => (
  <div className="loading-indicator">
    <div className="spinner"></div>
    <p>{status}</p>
  </div>
);

const MessageRenderer: FC<{ message: Message }> = ({ message }) => {
  return (
    <>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          code(props) {
            const {children, className, ...rest} = props
            return <CodeBlock>{String(children)}</CodeBlock>
          }
        }}
      >
        {message.parts.map(part => 'text' in part ? part.text : '').join('')}
      </ReactMarkdown>
      {message.parts.filter(part => 'inlineData' in part).map((part, index) => (
          'inlineData' in part && part.inlineData.mimeType.startsWith('image/') && (
          <img
            key={index}
            src={`data:${part.inlineData.mimeType};base64,${part.inlineData.data}`}
            alt="Generated content"
            className="generated-image"
          />
        )
      ))}
      {message.citations && message.citations.length > 0 && (
          <div className="citations">
              <h4 className="citation-title">Sources:</h4>
              {message.citations.map((citation, index) => (
                  <a key={index} href={citation.web.uri} target="_blank" rel="noopener noreferrer" className="citation-link">
                      {`[${index + 1}] ${citation.web.title}`}
                  </a>
              ))}
          </div>
      )}
    </>
  );
}

// -- MODAL COMPONENTS --

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}

const Modal: FC<ModalProps> = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">{title}</h2>
          <button className="icon-button modal-close-button" onClick={onClose} aria-label="Close modal">
            <CloseIcon />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
};

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  apiKey: string;
  setApiKey: (key: string) => void;
}

const SettingsModal: FC<SettingsModalProps> = ({ isOpen, onClose, apiKey, setApiKey }) => {
  const [localApiKey, setLocalApiKey] = useState(apiKey);

  const handleSave = () => {
    setApiKey(localApiKey);
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Settings">
      <div className="modal-form">
        <label htmlFor="apiKey">Gemini API Key</label>
        <input
          id="apiKey"
          type="password"
          className="modal-input"
          value={localApiKey}
          onChange={(e) => setLocalApiKey(e.target.value)}
          placeholder="Enter your API key"
        />
        <p className="modal-description">
          Your API key is stored securely in your browser's local storage and is never sent to any server other than Google's.
        </p>
        <div className="modal-actions">
          <button className="button button-secondary" onClick={onClose}>Cancel</button>
          <button className="button button-primary" onClick={handleSave}>Save</button>
        </div>
      </div>
    </Modal>
  );
};

interface AssistantsModalProps {
    isOpen: boolean;
    onClose: () => void;
    assistants: Assistant[];
    onSave: (assistant: Assistant | Partial<Assistant>) => void;
    onDelete: (id: string) => void;
    onSelect: (id: string) => void;
}

const AssistantsModal: FC<AssistantsModalProps> = ({ isOpen, onClose, assistants, onSave, onDelete, onSelect }) => {
    const [view, setView] = useState<'gallery' | 'form'>('gallery');
    const [editingAssistant, setEditingAssistant] = useState<Assistant | Partial<Assistant> | null>(null);

    const handleEdit = (assistant: Assistant) => {
        setEditingAssistant(assistant);
        setView('form');
    };

    const handleCreateNew = () => {
        setEditingAssistant({ name: '', description: '', systemInstruction: '', defaultMode: 'superfast' });
        setView('form');
    };

    const handleBackToGallery = () => {
        setEditingAssistant(null);
        setView('gallery');
    };

    const handleSave = (e: FormEvent) => {
        e.preventDefault();
        if (editingAssistant) {
            onSave(editingAssistant);
            handleBackToGallery();
        }
    };

    const handleFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        if (editingAssistant) {
            setEditingAssistant({ ...editingAssistant, [e.target.name]: e.target.value });
        }
    };
    
    const title = view === 'form' 
      ? (editingAssistant && 'id' in editingAssistant ? 'Edit Assistant' : 'Create Assistant') 
      : 'Assistants';
      
    return (
        <Modal isOpen={isOpen} onClose={onClose} title={title}>
            {view === 'gallery' ? (
                <div>
                    <div className="modal-actions" style={{justifyContent: 'flex-start', marginBottom: '1rem'}}>
                        <button className="button button-primary" onClick={handleCreateNew}><PlusIcon /> Create New</button>
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
                                    <button className="button button-primary" onClick={() => onSelect(assistant.id)}>Chat</button>
                                    <button className="icon-button" onClick={() => handleEdit(assistant)} aria-label="Edit"><EditIcon /></button>
                                    {assistant.id !== 'default' && 
                                        <button className="icon-button" onClick={() => onDelete(assistant.id)} aria-label="Delete"><DeleteIcon /></button>
                                    }
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            ) : (
                <form onSubmit={handleSave} className="modal-form">
                    <label htmlFor="name">Name</label>
                    <input id="name" name="name" className="modal-input" value={editingAssistant?.name || ''} onChange={handleFormChange} required />
                    
                    <label htmlFor="description">Description</label>
                    <input id="description" name="description" className="modal-input" value={editingAssistant?.description || ''} onChange={handleFormChange} />
                    
                    <label htmlFor="systemInstruction">System Instruction</label>
                    <textarea id="systemInstruction" name="systemInstruction" className="modal-textarea" value={editingAssistant?.systemInstruction || ''} onChange={handleFormChange} required />

                    <label htmlFor="defaultMode">Default Mode</label>
                    <select id="defaultMode" name="defaultMode" className="modal-input" value={editingAssistant?.defaultMode || 'superfast'} onChange={handleFormChange}>
                        <option value="superfast">Superfast</option>
                        <option value="goat">Goat</option>
                        <option value="ultra">Ultra</option>
                        <option value="super">Super</option>
                        <option value="image-gen">Image Gen</option>
                    </select>

                    <div className="modal-actions">
                        <button type="button" className="button button-secondary" onClick={handleBackToGallery}>Cancel</button>
                        <button type="submit" className="button button-primary">Save</button>
                    </div>
                </form>
            )}
        </Modal>
    );
};

// --- MAIN APP ---

const App: FC = () => {
  // State
  const [apiKey, setApiKey] = useLocalStorage<string>('gemini-api-key', '');
  const [assistants, setAssistants] = useLocalStorage<Assistant[]>('assistants', [DEFAULT_ASSISTANT]);
  const [chats, setChats] = useLocalStorage<ChatSession[]>('chats', []);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [userInput, setUserInput] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [mode, setMode] = useState<AppMode>('superfast');
  const [useGoogleSearch, setUseGoogleSearch] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isAssistantModalOpen, setIsAssistantModalOpen] = useState(false);
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  
  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const messageListRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Derived state
  const currentChat = chats.find(chat => chat.id === currentChatId);
  const currentAssistant = assistants.find(a => a.id === currentChat?.assistantId) || DEFAULT_ASSISTANT;

  // Effects
  useEffect(() => {
    if (!apiKey) {
      setIsSettingsModalOpen(true);
    }
  }, [apiKey]);

  useEffect(() => {
    // Migrate assistants to include defaultMode for backward compatibility
    setAssistants(prev => prev.map(a => ({
        ...a,
        defaultMode: a.defaultMode || 'superfast'
    })));
  }, []);
  
  useEffect(() => {
    if (!currentChatId && chats.length > 0) {
      setCurrentChatId(chats[0].id);
    } else if (chats.length === 0) {
        handleNewChat();
    }
  }, [chats, currentChatId]);
  
  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [currentChat?.messages, isLoading]);

  useEffect(() => {
    if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
        textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [userInput]);

  // Handlers
  const handleNewChat = (assistantId: string = DEFAULT_ASSISTANT.id) => {
    const selectedAssistant = assistants.find(a => a.id === assistantId) || DEFAULT_ASSISTANT;
    const newChat: ChatSession = {
      id: uuidv4(),
      title: 'New Chat',
      messages: [],
      assistantId,
    };
    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    setMode(selectedAssistant.defaultMode);
    setIsSidebarOpen(false);
    setIsAssistantModalOpen(false);
  };

  const handleDeleteChat = (chatId: string) => {
    setChats(chats.filter(chat => chat.id !== chatId));
    if (currentChatId === chatId) {
      setCurrentChatId(chats.length > 1 ? chats.filter(c => c.id !== chatId)[0].id : null);
    }
  };

  const handleAttachment = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files).map(file => ({
        file,
        previewUrl: URL.createObjectURL(file)
      }));
      setAttachments(prev => [...prev, ...files]);
    }
  };
  
  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new MediaRecorder(stream);
        audioChunksRef.current = [];
        mediaRecorderRef.current.ondataavailable = event => {
          audioChunksRef.current.push(event.data);
        };
        mediaRecorderRef.current.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          const audioFile = new File([audioBlob], "voice_recording.webm", { type: 'audio/webm' });
          const attachment = { file: audioFile, previewUrl: URL.createObjectURL(audioFile) };
          setAttachments(prev => [...prev, attachment]);
          stream.getTracks().forEach(track => track.stop());
        };
        mediaRecorderRef.current.start();
        setIsRecording(true);
      } catch (error) {
        console.error("Error accessing microphone:", error);
        alert("Could not access microphone. Please check permissions.");
      }
    }
  };
  
  const handleSaveAssistant = (assistantData: Assistant | Partial<Assistant>) => {
    if ('id' in assistantData && assistantData.id) {
        setAssistants(prev => prev.map(a => a.id === assistantData.id ? { ...a, ...assistantData } as Assistant : a));
    } else {
        const newAssistant: Assistant = {
            id: uuidv4(),
            name: assistantData.name || 'New Assistant',
            description: assistantData.description || '',
            systemInstruction: assistantData.systemInstruction || '',
            avatar: (assistantData.name || 'N')[0].toUpperCase(),
            defaultMode: (assistantData as Assistant).defaultMode || 'superfast',
        };
        setAssistants(prev => [newAssistant, ...prev]);
    }
  };
  
  const handleDeleteAssistant = (assistantId: string) => {
      if (assistantId === 'default') return;
      if (chats.some(c => c.assistantId === assistantId)) {
          alert("Cannot delete an assistant that is being used in a chat.");
          return;
      }
      setAssistants(prev => prev.filter(a => a.id !== assistantId));
  };

  const addMessage = (chatId: string, message: Message) => {
    setChats(prevChats => prevChats.map(chat =>
      chat.id === chatId ? { ...chat, messages: [...chat.messages, message] } : chat
    ));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const userInputText = userInput.trim();
    if (!userInputText && attachments.length === 0) return;
    if (!currentChatId || !currentChat) return;

    if (!apiKey) {
      setIsSettingsModalOpen(true);
      alert("Please set your Gemini API key in Settings before sending a message.");
      return;
    }

    setIsLoading(true);
    setLoadingStatus('Preparing request...');

    // Create user message object from inputs
    const userParts: Part[] = [];
    if (userInputText) {
      userParts.push({ text: userInputText });
    }
    const attachmentParts = await Promise.all(
      attachments.map(att => fileToGenerativePart(att.file))
    );
    userParts.push(...attachmentParts);
    const hasImageAttachment = attachments.some(att => att.file.type.startsWith('image/'));
    const userMessage: Message = { id: uuidv4(), role: 'user', parts: userParts };

    // Prepare data for API calls using state *before* any updates
    const isFirstMessage = currentChat.messages.length === 0;
    const capturedUserInput = userInput;
    const chatHistoryForAPI = currentChat.messages.map(msg => ({
      role: msg.role as 'user' | 'model',
      parts: msg.parts,
    }));
    
    // Clear inputs immediately for a responsive feel
    setUserInput('');
    setAttachments([]);

    try {
      const ai = new GoogleGenAI({ apiKey });
      let finalTitle = currentChat.title;

      // If it's the first message, generate the title *before* updating the state.
      if (isFirstMessage && capturedUserInput.trim()) {
        setLoadingStatus('Generating title...');
        const titleResponse = await ai.models.generateContent({
            model: CHAT_MODEL,
            contents: [{role: 'user', parts: [{text: capturedUserInput.trim()}]}],
            config: { systemInstruction: TITLE_GEN_INSTRUCTION }
        });
        finalTitle = titleResponse.text.trim().replace(/"/g, '');
      }
      
      // Perform a single, combined state update for the user message and the new title.
      // This prevents the race condition that was causing the user's message to disappear.
      setChats(prevChats =>
        prevChats.map(chat =>
          chat.id === currentChatId
            ? {
                ...chat,
                title: finalTitle,
                messages: [...chat.messages, userMessage],
              }
            : chat
        )
      );
      
      const currentUserTurn: Content = { role: 'user', parts: userMessage.parts };
      let finalResponseParts: Part[] = [];
      let citations: any[] | undefined = undefined;

      // --- Main Logic ---
      if (hasImageAttachment) {
          setLoadingStatus('Analyzing image...');
          const result = await ai.models.generateContent({
              model: IMAGE_EDIT_MODEL,
              contents: [{ role: 'user', parts: userParts }],
              config: {
                  responseModalities: [Modality.IMAGE, Modality.TEXT],
              },
          });
          finalResponseParts = result.candidates?.[0]?.content?.parts || [];
      } 
      else if (mode === 'image-gen') {
        setLoadingStatus('Generating image...');
        const response = await ai.models.generateImages({
            model: IMAGE_GEN_MODEL,
            prompt: userInputText,
            config: { numberOfImages: 1 },
        });
        const imagePart: Part = { inlineData: { mimeType: 'image/png', data: response.generatedImages[0].image.imageBytes } };
        finalResponseParts = [imagePart];
      }
      else {
          if (mode === 'superfast') {
            setLoadingStatus('Thinking...');
            const result = await ai.models.generateContent({
              model: CHAT_MODEL,
              contents: [...chatHistoryForAPI, currentUserTurn],
              config: { 
                systemInstruction: currentAssistant.systemInstruction,
                thinkingConfig: { thinkingBudget: 0 },
                ...(useGoogleSearch && { tools: [{googleSearch: {}}]})
              },
            });
            finalResponseParts = result.candidates?.[0]?.content?.parts || [];
            citations = result.candidates?.[0]?.groundingMetadata?.groundingChunks;
          } else { // Multi-agent modes
            setLoadingStatus('Initializing agents...');
            const initialAgentPromises = Array(4).fill(0).map(() =>
              ai.models.generateContent({
                model: CHAT_MODEL, contents: [...chatHistoryForAPI, currentUserTurn], config: { systemInstruction: INITIAL_SYSTEM_INSTRUCTION },
              })
            );
            const initialResponses = await Promise.all(initialAgentPromises);
            const initialAnswers = initialResponses.map(res => res.text);
      
            setLoadingStatus('Refining answers...');
            const refinementAgentPromises = initialAnswers.map((initialAnswer, index) => {
              const otherAnswers = initialAnswers.filter((_, i) => i !== index);
              const refinementContext = `My initial response was: "${initialAnswer}". The other agents responded with: 1. "${otherAnswers[0]}" 2. "${otherAnswers[1]}" 3. "${otherAnswers[2]}". Re-evaluate and provide an improved response.`;
              const refinementTurn: Content = { role: 'user', parts: [{ text: `${userInputText}\n\n---INTERNAL CONTEXT---\n${refinementContext}` }] };
              return ai.models.generateContent({ model: CHAT_MODEL, contents: [refinementTurn], config: { systemInstruction: REFINEMENT_SYSTEM_INSTRUCTION } });
            });
            const refinedResponses = await Promise.all(refinementAgentPromises);
            const refinedAnswers = refinedResponses.map(res => res.text);
      
            setLoadingStatus('Synthesizing final response...');
            const synthesizerContext = `Synthesize these four refined responses into the single best final answer for the user's query: "${userInputText}"\n\n1: "${refinedAnswers[0]}"\n2: "${refinedAnswers[1]}"\n3: "${refinedAnswers[2]}"\n4: "${refinedAnswers[3]}"`;
            const synthesizerTurn: Content = { role: 'user', parts: [{ text: synthesizerContext }] };
      
            const finalResult = await ai.models.generateContent({ model: CHAT_MODEL, contents: [synthesizerTurn], config: { systemInstruction: SYNTHESIZER_SYSTEM_INSTRUCTION } });
            finalResponseParts = finalResult.candidates?.[0]?.content?.parts || [];
          }
      }
      
      const finalMessage: Message = { id: uuidv4(), role: 'model', parts: finalResponseParts, citations };
      addMessage(currentChatId, finalMessage);

    } catch (error) {
      console.error('Error generating content:', error);
      const errorMessageText = error instanceof Error ? error.message : 'Sorry, an error occurred. Please try again.';
      const errorMessage: Message = { id: uuidv4(), role: 'model', parts: [{ text: `Error: ${errorMessageText}` }] };
      addMessage(currentChatId, errorMessage);
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  return (
    <div className="app-container">
      <div className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
            <h1 className="sidebar-title">Chats</h1>
            <button className="icon-button new-chat-button" onClick={() => handleNewChat()} aria-label="New Chat"><PlusIcon /></button>
        </div>
        <div className="chat-list">
            {chats.map(chat => (
                <div key={chat.id} className={`chat-list-item ${currentChatId === chat.id ? 'active' : ''}`} onClick={() => {setCurrentChatId(chat.id); setIsSidebarOpen(false);}}>
                    <span className="chat-item-title">{chat.title}</span>
                    <button className="delete-chat-button" onClick={(e) => { e.stopPropagation(); handleDeleteChat(chat.id); }}><DeleteIcon /></button>
                </div>
            ))}
        </div>
        <div className="sidebar-footer">
            <button className="button button-secondary" onClick={() => setIsAssistantModalOpen(true)}>Assistants</button>
            <button className="icon-button" onClick={() => setIsSettingsModalOpen(true)} aria-label="Settings"><SettingsIcon /></button>
        </div>
      </div>

      <main className="chat-area">
        <header className="chat-header">
            <button className="icon-button mobile-menu-button" onClick={() => setIsSidebarOpen(!isSidebarOpen)}><MenuIcon /></button>
            <div className='chat-title-group'>
              <div className="avatar small-avatar" aria-label={`${currentAssistant.name} avatar`}>{currentAssistant.avatar}</div>
              <h2 className="chat-title">{currentChat?.title || 'Chat'}</h2>
            </div>
            <div className="mode-selector">
                <select value={mode} onChange={e => setMode(e.target.value as AppMode)} disabled={isLoading}>
                    <option value="superfast">Superfast</option>
                    <option value="goat">Goat</option>
                    <option value="ultra">Ultra</option>
                    <option value="super">Super</option>
                    <option value="image-gen">Image Gen</option>
                </select>
            </div>
        </header>

        <div className="message-list" ref={messageListRef}>
            {currentChat?.messages.map(msg => (
              <div key={msg.id} className={`message-container ${msg.role}`}>
                <div className="avatar" aria-label={`${msg.role} avatar`}>
                  {msg.role === 'user' ? 'U' : currentAssistant.avatar}
                </div>
                <div className="message-bubble">
                   <div className="message-content">
                     <MessageRenderer message={msg} />
                   </div>
                </div>
              </div>
            ))}
            {isLoading && <LoadingIndicator status={loadingStatus} />}
        </div>
        
        <div className="input-area-container">
          {attachments.length > 0 && (
            <div className="attachments-preview">
              {attachments.map((att, index) => (
                <div key={index} className="attachment-item">
                  {att.file.type.startsWith('image/') ? (
                    <img src={att.previewUrl} alt={att.file.name} />
                  ) : (
                    <div className="file-placeholder"><FileIcon /><span style={{fontSize:'10px', overflow:'hidden', textOverflow:'ellipsis'}}>{att.file.name}</span></div>
                  )}
                  <button className="remove-attachment-button" onClick={() => removeAttachment(index)}>&times;</button>
                </div>
              ))}
            </div>
          )}
          <form onSubmit={handleSubmit} className="input-form">
            <label htmlFor="file-upload" className="icon-button" aria-label="Attach file">
                <AttachmentIcon />
            </label>
            <input id="file-upload" type="file" multiple onChange={handleAttachment} style={{ display: 'none' }} disabled={isLoading} />
            <button type="button" className={`icon-button ${isRecording ? 'recording' : ''}`} onClick={toggleRecording} disabled={isLoading} aria-label="Record voice"><MicIcon /></button>
            <textarea
              ref={textareaRef}
              className="input-field"
              value={userInput}
              onChange={e => setUserInput(e.target.value)}
              placeholder={`Message ${currentAssistant.name}...`}
              rows={1}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e as any);
                }
              }}
              disabled={isLoading || !apiKey}
            />
            <button type="submit" className="icon-button send-button" disabled={isLoading || !apiKey} aria-label="Send message"><SendIcon /></button>
          </form>
           <div style={{display: 'flex', justifyContent: 'center', paddingTop: '8px'}}>
                <label className="search-toggle">
                    <span className="search-toggle-label">Google Search</span>
                    <div className="switch">
                        <input type="checkbox" checked={useGoogleSearch} onChange={() => setUseGoogleSearch(!useGoogleSearch)} />
                        <span className="slider"></span>
                    </div>
                </label>
           </div>
        </div>
      </main>
      
      <SettingsModal 
        isOpen={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        apiKey={apiKey}
        setApiKey={setApiKey}
      />
      
      <AssistantsModal
        isOpen={isAssistantModalOpen}
        onClose={() => setIsAssistantModalOpen(false)}
        assistants={assistants}
        onSave={handleSaveAssistant}
        onDelete={handleDeleteAssistant}
        onSelect={handleNewChat}
      />
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
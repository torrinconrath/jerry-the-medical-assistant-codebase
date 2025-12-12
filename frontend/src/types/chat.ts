export interface ChatMessage {
  type: "text" | "bot" | "file" | "audio";
  content: string | File;
  previewUrl?: string;
  timestamp?: string;

  fileName?: string; 
  decodedText?: string;
  isProcessed?: boolean;

  isRecording?: boolean;
}
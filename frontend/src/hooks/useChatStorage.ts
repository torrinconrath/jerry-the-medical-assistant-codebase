import { useState, useEffect } from "react";
import type { ChatMessage } from "../types/chat";

const STORAGE_KEY = "chatbot_messages";

export function useChatStorage() {
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch (err) {
      console.error("Failed to parse chat history from localStorage:", err);
      return [];
    }
  });

  // Save messages whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch (err) {
      console.error("Failed to save chat history:", err);
    }
  }, [messages]);

  // Optionally expose a clear function
  const clearMessages = () => {
    setMessages([]);
    localStorage.removeItem(STORAGE_KEY);
  };

  return { messages, setMessages, clearMessages };
}

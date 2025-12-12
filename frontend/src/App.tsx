import React, { useState, useRef, useEffect } from "react";
import FileDecoder from "./components/FileDecoder";
import VoiceOutput from "./components/VoiceOutput";
import DevMetrics from "./components/DevMetrics";
import type { ChatMessage } from "./types/chat";
import { createVoiceDecoder, type VoiceRecognitionHandlers } from "./components/VoiceDecoder";
import { useChatStorage } from "./hooks/useChatStorage";
import { useDevMetrics } from "./hooks/useDevMetrics";
import ReactMarkdown from "react-markdown";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash, faWrench, faCopy, faPaperPlane, faFile, faFileAlt, faMicrophone, faStop } from '@fortawesome/free-solid-svg-icons';
import chatbotIcon from "./assets/medical_chatbot_icon.png";
import "./App.css";

function App() {
  const { messages, setMessages, clearMessages } = useChatStorage();
  const [input, setInput] = useState("");
  const [listening, setListening] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const voiceRef = useRef<VoiceRecognitionHandlers | null>(null);
  const chatWindowRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(false);

  const MAX_INPUT_CHARS = 10000;
  const BASE_URL = import.meta.env.VITE_SFT_MODEL_ENDPOINT;


  // Simplified DevMetrics hook (no callbacks)
  const { metrics, startMonitoring, completeMonitoring, clearMetrics } = useDevMetrics(devMode);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }

    // console.log(messages);
  }, [messages]);

  // Dev mode function
  const toggleDevMode = () => setDevMode((prev) => !prev);


  // ----------------------------------------------------------------
  // Text Sending Logic
  // ----------------------------------------------------------------
  const handleSendText = async () => {
    if (!input.trim()) return;

    const userMessage: ChatMessage = {
      type: "text",
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    await sendMessage(input);
    setInput("");
  };

  // ----------------------------------------------------------------
  // File Sending Logic
  // ----------------------------------------------------------------
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const fileUrl = URL.createObjectURL(file);
      const fileMessage: ChatMessage = {
        type: "file",
        content: file,
        previewUrl: fileUrl,
        timestamp: new Date().toISOString(),
        fileName: file.name,
        isProcessed: false, // Initialize the file as not processed
      };
      setMessages((prev) => [...prev, fileMessage]);
    }
    
    // Reset the input value
    e.target.value = '';
  };

  // Fetches the handles the file after decoding
  const handleFileDecoded = async (timestamp: string, decodedText: string) => {
    
    // Set the message to processed 
    setMessages(prevMessages =>
      prevMessages.map(msg =>
        msg.timestamp === timestamp 
          ? { ...msg, decodedText: decodedText.trim(), isProcessed: true } : msg
      )
    );
 
    if (decodedText && decodedText !== "No text could be recognized." && !decodedText.includes("Error decoding")) {
      const filePrompt = "Observe the contents of the text: " + decodedText;
      await sendMessage(filePrompt);
    }
  };

  // ----------------------------------------------------------------
  // Voice Sending Logic
  // ----------------------------------------------------------------
  const handleVoiceToggle = () => {
    if (!voiceRef.current) {
      voiceRef.current = createVoiceDecoder(

        // Handles live transcript
        (transcript) => {
          setMessages((prev) => {
            const newMessages = [...prev];

            const activeIndex = newMessages.findIndex(
              (msg) => msg.type === "audio" && msg.isRecording === true
            );
            
            if (activeIndex !== -1) {
              // Update only the currently active recording message
              newMessages[activeIndex] = {
                ...newMessages[activeIndex],
                content: transcript,
              };
              return newMessages;
            }

            return [
              ...prev,
              { type: "audio", content: transcript, timestamp: new Date().toISOString(), isRecording: true},
            ];
          });
        },

        // Handles the recording stop
        () => {
          setListening(false);
          if (voiceRef.current) {
            const finalText = voiceRef.current.getDecodedText();

            if (finalText.trim()) {

              setMessages((prev) =>
                prev.map((msg) => 
                  msg.type === "audio" && msg.isRecording 
                    ? { ...msg, content: finalText, isRecording: false} 
                    : msg
                )
              );
              sendMessage(finalText);         

            } else {

              // Mark recording ended (no usable transcript)
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.type === "audio" && msg.isRecording
                    ? { ...msg, isRecording: false }
                    : msg
                )
              );
            }
          }
        }
      );
      if (!voiceRef.current) return;
    }

    if (listening) {
      voiceRef.current.stop();
      setListening(false);
    } else {
      voiceRef.current.start();
      setListening(true);
    }
  };

  // ----------------------------------------------------------------
  //  Send Message Function with Streaming
  // ----------------------------------------------------------------
  const sendMessage = async ( content: string ) => {
    if (!content.trim()) return;

    // Message if the input exceeds the safe limit
    if (content.length > MAX_INPUT_CHARS) {
      const errorMessage: ChatMessage = {
        type: "bot",
        content: `Your message is too long. Please reduce it to under ${MAX_INPUT_CHARS.toLocaleString()} characters.`,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
      return;
    }

    // Start metrics collection (only if devMode is active)
    if (devMode) startMonitoring();
    setLoading(true)

    // Set response to empty
    const responseTimestamp = new Date().toISOString();
    const response: ChatMessage = {
      type: "bot",
      content:
        "",
      timestamp: responseTimestamp,
    };
    setMessages((prev) => [...prev, response]);

    try {

      // Connect to streaming endpoint
      const res = await fetch(
        `${BASE_URL}/chat/stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: content }),
        }
      );

      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      if (!res.body) throw new Error("Response body is null");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      let fullResponseText = "";
      let inferenceTime = 0;

      // Reading loop
      while (true){
        const {done, value} = await reader.read();
        if (done) break;

        // Decode chunk and append it
        buffer += decoder.decode(value, {stream: true});

        // Process lines
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          try {
            const data = JSON.parse(line);

            // Update the content 
            if (data.type === "content") {
              fullResponseText += data.text;

              // Update response message
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.timestamp === responseTimestamp && msg.type === "bot"
                    ? { ...msg, content: fullResponseText }
                    : msg
                )
              );
            }

            // Update the inference time
            else if (data.type === "usage") {
              inferenceTime = data.inference_time;
            }

            // Server Error
            else if (data.type === "error") {
               console.error("Stream error:", data.message);
               fullResponseText += ` [Error: ${data.message}]`;
            }

          } catch (e) {
            console.warn("JSON Parse Error", e);
          }
        }
      }

      // In case the model cannot give a good response
      if (fullResponseText.trim().length < 4) {
        const errorMessage: ChatMessage = {
          type: "bot",
          content:
            "Sorry, the provided text is unable to be processed. Please rephrase it and try again.",
          timestamp: new Date().toISOString(),
        };

        // Replace the response with the error
        setMessages((prev) => 
            prev.map(msg => msg.timestamp === responseTimestamp && msg.type === "bot" ? errorMessage : msg)
        );
        return;
      }

      // Complete metrics with inference time (if available)
      if (devMode) completeMonitoring(fullResponseText, inferenceTime);


    } catch (err) {
      console.error("Error talking to backend:", err);
      const errorMessage: ChatMessage = {
        type: "bot",
        content:
          "Sorry, I'm having trouble connecting right now. Please try again.",
        timestamp: new Date().toISOString(),
      };
      
      // Replace response with error
      setMessages((prev) => 
        prev.map(msg => msg.timestamp === responseTimestamp && msg.type === "bot" ? errorMessage : msg)
      );

      // Complete metrics with error
      if (devMode) completeMonitoring("Error response", 0);

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <img src={chatbotIcon} alt="Chatbot Icon" className="chatbot-icon" />
          <h1>Jerry the Medical Assistant</h1>
          <button
            className="toggle-button"
            onClick={clearMessages}
            title="Clear chat history"
          >
            <FontAwesomeIcon icon={faTrash} /> Clear Chat
          </button>

          <button
            className={`toggle-button ${devMode ? "dev-active" : ""}`}
            onClick={toggleDevMode}
            title="Toggle development metrics"
          >
            <FontAwesomeIcon icon={faWrench} /> {devMode ? "Dev Mode ON" : "Dev Mode"}
          </button>
        </div>
      </header>

      <div className="main-content">
        <div className="chat-section">
          <div className="chat-window" ref={chatWindowRef}>
            {messages.length === 0 && (
              <div className="welcome-message">
                <p>Welcome! You can:</p>
                <ul>
                  <li>Type your message</li>
                  <li>Upload files (images/PDFs) for text extraction</li>
                  <li>Use voice input with the microphone</li>
                </ul>
                <p>Bot responses will include text-to-speech.</p>
                {devMode && (
                  <div className="dev-mode-notice">
                    Development metrics are enabled
                  </div>
                )}
              </div>
            )}

            {messages.map((msg, index) => (
              <div
                key={`${msg.timestamp}-${index}`}
                className={`chat-message ${
                  msg.type === "text" ||
                  msg.type === "file" ||
                  msg.type === "audio"
                    ? "user-message"
                    : msg.type === "bot"
                    ? "bot-message"
                    : ""
                }`}
              >
                <div className="message-content">
                  {msg.type === "text" && <span>{msg.content as string}</span>}
                  {msg.type === "bot" && (
                    <div className="bot-message-container">

                      {/* Show loading animation if there is no content */}
                      {(!msg.content || (typeof msg.content === 'string' && msg.content === '')) ? (
                        <div className="loading-message">
                          <div className="loading-dots">
                            <span>.</span>
                            <span>.</span>
                            <span>.</span>
                          </div>
                          <p>Bot is typing...</p>
                        </div>
                      ) : (
                        <>
                          {/* Else show content */}
                          <ReactMarkdown
                            components={{
                              strong: ({ node, ...props }) => <strong style={{ fontWeight: "bold" }} {...props} />,
                              li: ({ node, ...props }) => <li style={{ marginLeft: "1.2em" }} {...props} />,
                              ul: ({ node, ...props }) => <ul style={{ paddingLeft: "1.5em", marginTop: "0.5em" }} {...props} />,
                              p: ({ node, ...props }) => <p style={{ marginBottom: "0.5em" }} {...props} />,
                            }}
                          >
                            {msg.content as string}
                          </ReactMarkdown>

                          {msg.content && typeof msg.content === "string" && msg.content.trim() && (
                            <div className="voice-output-buttons">
                                <VoiceOutput text={msg.content as string} />
                                <button
                                  className="copy-button"
                                  onClick={() => navigator.clipboard.writeText(msg.content as string)}
                                >
                                  <FontAwesomeIcon icon={faCopy} /> Copy
                                </button>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  )}
                  {msg.type === "file" && (
                    <div className="file-message">

                      {/* File preview section */}
                      {msg.content instanceof File && msg.content.type.startsWith("image/") ? (
                        <img
                          src={msg.previewUrl}
                          alt={msg.content.name}
                          className="chat-image"
                        />
                      ) : (
                        <div className="file-info">
                          <FontAwesomeIcon icon={faFileAlt} /> <strong>{msg.content instanceof File ? msg.content.name : "Uploaded File"}</strong>
                        </div>
                      )}

                      {/* If decoded text already exists, just show it */}
                      {msg.decodedText && (
                        <div className="decoded-message">
                          <strong>Extracted text:</strong> {msg.decodedText}
                        </div>
                      )}
                      
                      {!msg.decodedText && (
                        <FileDecoder message={msg} onDecoded={handleFileDecoded} />
                      )}

                    </div>
                  )}
                  {msg.type === "audio" && (
                    <div className="audio-message">
                      <span> <FontAwesomeIcon icon={faMicrophone} /> {msg.content as string}</span>
                      {listening && (
                        <div className="recording-indicator">● Recording...</div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="input-area">
            <input
              name="chat-input"
              type="text"
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSendText()}
              placeholder="Type your message or use voice/file input..."
              disabled={loading} 
            />
            <button className="send-button" onClick={handleSendText} disabled={loading}>
              <FontAwesomeIcon icon={faPaperPlane} /> Send
            </button>

            <button className="send-button file-upload-button">
              <label>
                <FontAwesomeIcon icon={faFile} />  Upload
                <input
                  type="file"
                  onChange={handleFileUpload}
                  accept=".pdf,.png,.jpg,.jpeg,.gif"
                  hidden
                  disabled={loading}
                />
              </label>
            </button>

            <button
              className={`voice-button send-button ${listening ? "listening" : ""}`}
              onClick={handleVoiceToggle}
              disabled={loading}
            >
              {listening ? <FontAwesomeIcon icon={faStop} /> : <FontAwesomeIcon icon={faMicrophone} />} {listening ? "Stop" : "Voice"}
            </button>
          </div>
        </div>

        {/* Dev metrics panel */}
        <DevMetrics
          isEnabled={devMode}
          metrics={metrics}
          clearMetrics={clearMetrics}
        />
      </div>
    </div>
  );
}

export default App;

// Here are two normal medical questions you could ask a bot:

// What are the common symptoms of seasonal allergies and what over-the-counter treatments are usually recommended?

// When should someone seek medical attention for a persistent headache?

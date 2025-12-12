import React, { useState, useEffect } from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faVolumeUp, faVolumeMute } from '@fortawesome/free-solid-svg-icons';

interface VoiceOutputProps {
  text: string;
  autoPlay?: boolean;
}

const VoiceOutput: React.FC<VoiceOutputProps> = ({ text, autoPlay = false }) => {
  const [speaking, setSpeaking] = useState(false);
  const [supported, setSupported] = useState(true);

  useEffect(() => {
    if (!('speechSynthesis' in window)) {
      setSupported(false);
      return;
    }

    // Check if voices are available
    const checkVoices = () => {
      const voices = speechSynthesis.getVoices();
      setSupported(voices.length > 0);
    };

    checkVoices();
    speechSynthesis.onvoiceschanged = checkVoices;

    return () => {
      speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  const speak = () => {
    if (!text || !supported) return;
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;

    // Try to find a good voice
    const voices = speechSynthesis.getVoices();
    const preferredVoice = voices.find(voice => 
      voice.lang.includes('en') && voice.localService === true
    ) || voices.find(voice => voice.lang.includes('en'));
    
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };

  const stopSpeaking = () => {
    window.speechSynthesis.cancel();
    setSpeaking(false);
  };

  useEffect(() => {
    if (autoPlay && text && supported) {
      speak();
    }
    return () => {
      window.speechSynthesis.cancel();
    };
  }, [text, autoPlay, supported]);

  if (!supported) {
    return null; // Hide button if not supported
  }

  return (
    <div className="voice-output">
      <button
        onClick={speaking ? stopSpeaking : speak}
        disabled={!text}
        style={{ 
          backgroundColor: speaking ? "#ffcc00" : "#4CAF50",
          color: speaking ? "#000" : "#fff"
        }}
      >
        {speaking ? <FontAwesomeIcon icon={faVolumeMute} /> : <FontAwesomeIcon icon={faVolumeUp} />} {speaking ? "Stop" : "Speak"}
      </button>
    </div>
  );
};

export default VoiceOutput;
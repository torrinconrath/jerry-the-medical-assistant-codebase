export type VoiceRecognitionHandlers = {
  start: () => void;
  stop: () => void;
  getDecodedText: () => string;
};

export function createVoiceDecoder(
  onResult: (text: string) => void,
  onEnd?: () => void
): VoiceRecognitionHandlers | null {
  if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
    alert("❌ Speech recognition not supported in this browser.");
    return null;
  }

  const SpeechRecognition =
    (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  let decodedText = "";
  let isRunning = false;

  recognition.onresult = (event: any) => {
    let interimTranscript = '';
    let finalTranscript = '';

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript;
      } else {
        interimTranscript += event.results[i][0].transcript;
      }
    }

    if (finalTranscript) {
      decodedText += finalTranscript + ' ';
      onResult(decodedText.trim());
    } else if (interimTranscript) {
      onResult(decodedText + interimTranscript);
    }
  };

  recognition.onerror = (event: any) => {
    console.error('Speech recognition error:', event.error);
    if (event.error === 'no-speech') {
      onResult(decodedText.trim());
    }
  };

  recognition.onend = () => {
    isRunning = false;
    if (onEnd) onEnd();
  };

  return {
    start: () => {
      if (!isRunning) {
        decodedText = "";
        isRunning = true;
        recognition.start();
      }
    },
    stop: () => {
      if (isRunning) {
        isRunning = false;
        recognition.stop();
      }
    },
    getDecodedText: () => decodedText.trim(),
  };
}
import React, { useEffect, useRef, useState } from "react";
import { createWorker, PSM } from "tesseract.js";
import * as pdfjsLib from "pdfjs-dist";
import type { ChatMessage } from "../types/chat";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSpinner } from '@fortawesome/free-solid-svg-icons';

// ✅ Worker import for bundlers like Vite/CRA
import pdfWorker from "pdfjs-dist/build/pdf.worker?url";
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;

interface FileDecoderProps {
  message: ChatMessage;
  onDecoded: (timestamp: string, text: string) => void;
}

const FileDecoder: React.FC<FileDecoderProps> = ({ message, onDecoded }) => {
  const [decoded, setDecoded] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const hasDecodedRef = useRef(false); 

  useEffect(() => {
    const processFile = async () => {
      if (hasDecodedRef.current ||message.isProcessed || message.type !== "file") return;
      hasDecodedRef.current = true; 

      setLoading(true);
      const file = message.content as File;

      try {
        let extractedText = "";

        if (file.type === "application/pdf") {
          // Process PDF
          const url = URL.createObjectURL(file);
          const pdf = await pdfjsLib.getDocument(url).promise;
          let fullText = "";

          const worker = await createWorker("eng");
          await worker.setParameters({
            tessedit_pageseg_mode: PSM.AUTO,
          });

          // Process only first 5 pages 
          const pageLimit = Math.min(pdf.numPages, 5);
          for (let i = 1; i <= pageLimit; i++) {

            const page = await pdf.getPage(i);
            const viewport = page.getViewport({ scale: 1.5 });

            const canvas = document.createElement("canvas");
            const context = canvas.getContext("2d")!;
            canvas.width = viewport.width;
            canvas.height = viewport.height;

            // Render parameters
            await page.render({
              canvasContext: context,
              viewport: viewport,
              canvas: canvas, 
            }).promise;

            const { data } = await worker.recognize(canvas);
            fullText += data.text + "\n\n";
          }

          await worker.terminate();
          URL.revokeObjectURL(url);
          extractedText = fullText.replace(/\s+/g, " ").trim();

        } else if (file.type.startsWith("image/")) {

          const worker = await createWorker("eng");
          await worker.setParameters({
            tessedit_pageseg_mode: PSM.AUTO,
          });

          // Process image
          const url = URL.createObjectURL(file);

          const { data } = await worker.recognize(url);
          extractedText = data.text.replace(/\s+/g, " ").trim();
          
          await worker.terminate();
          URL.revokeObjectURL(url);
        } else {
          extractedText = "Unsupported file type";
        }

        if (!extractedText) {
          extractedText = "No text could be recognized in this file.";
        }

        setDecoded(extractedText);
        if (!message.isProcessed && message.timestamp) {
          onDecoded(message.timestamp, extractedText);
        }
      } catch (err) {
        console.error("File processing failed:", err);
        const errorText = "Error processing file. Please try another file.";
        setDecoded(errorText);

        if (!message.isProcessed && message.timestamp) {
          onDecoded(message.timestamp, errorText);
        }
      } finally {
        setLoading(false);
      }
    };

    processFile();
  }, [message.timestamp, message.isProcessed]);

  if (loading) {
    return <div className="decoded-message"><FontAwesomeIcon icon={faSpinner} spin /> Processing file...</div>;
  }

  return decoded ? (
    <div className="decoded-message">
      <strong>Extracted text:</strong> {decoded}
    </div>
  ) : null;
};

export default FileDecoder;
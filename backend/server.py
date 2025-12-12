from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from vllm import SamplingParams
import nest_asyncio
import uvicorn
from pyngrok import ngrok
import torch, gc, os, re, time, uuid, json
from typing import Optional
from fastapi.responses import StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from contextlib import asynccontextmanager

# ----------------------------
# Apply asyncio patch
# ----------------------------
nest_asyncio.apply()

# ----------------------------
# Environment variables
# ----------------------------
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTHTOKEN")
DEV_DOMAIN = os.getenv("DEV_DOMAIN")
PORT = 5713
MODEL_PATH = os.path.expanduser("~/models/SFT_merged_model_awq")

public_url = None
llm_engine = None
sampling_params = None

# ----------------------------
# Lifespan manager
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    global llm_engine, sampling_params, public_url
    print("Starting Up...")

    # ----------------------------
    # Ngrok setup
    # ----------------------------
    if NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)

        # Close any existing tunnels to avoid duplicates
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)

        # Open a new tunnel with your reserved domain
        try:
            tunnel = ngrok.connect(addr=PORT, proto="http", domain=DEV_DOMAIN)
            public_url = tunnel.public_url
            print(f"✅ Ngrok tunnel active: {public_url} → localhost:{PORT}")

        except Exception as e:
            print(f"❌ Failed to start ngrok tunnel: {e}")
    else:
        print("⚠️ Warning: NGROK_AUTHTOKEN not set. Public tunnel disabled.")

    # ----------------------------
    # Model setup
    # ----------------------------

    # Clear GPU memory before loading
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading model with vLLM Async Engine...")

    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        max_model_len=4096,
        dtype="float16",
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        quantization="awq_marlin",
    )

    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=2048,
        top_p=0.9,
        skip_special_tokens=True,
    )

    print("Model loaded successfully!")
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ----------------------------
    # Lifespan Processes
    # ----------------------------

    # Running
    yield

    # Shutdown
    print("Shutting down...")
    if llm_engine:
        del llm_engine

    gc.collect()
    torch.cuda.empty_cache()

    ngrok.kill()
    print("Shutdown complete!")


# ----------------------------
# FastAPI app setup
# ----------------------------
app = FastAPI(lifespan=lifespan) # runs app with lifespan

# Allow all origins for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request/response schemas
# ----------------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    inference_time: Optional[float] = None

# ----------------------------
# Helper Functions
# ----------------------------

def format_prompt(user_message):
    return (
        "<|im_start|>system\n"
        "You are a medical assistant. Provide clear, factual answers about health topics. "
        "Always explain key medical terms in plain English. Be concise but understandable.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_message}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

async def stream_processor(prompt: str, request_id: str):

    if llm_engine is None:
        yield json.dumps({"type": "error", "message": "Model still loading..."}) + "\n"
        return

    results_generator = llm_engine.generate(prompt, sampling_params, request_id)

    start_time = time.time()
    previous_text = ""

    # Set chain-of-thought pattern
    think_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
    try:
        async for request_output in results_generator:
            full_text = request_output.outputs[0].text

            # Check if the model is still thinking
            is_thinking = "<think>" in full_text and "</think>" not in full_text

            if is_thinking:
                # Send ping to keep the connection alive
                yield json.dumps({"type": "ping"}) + "\n"

            else:
                # Remove thoughts
                cleaned_text = re.sub(think_pattern, "", full_text)

                # Send only the new characters
                if len(cleaned_text) > len(previous_text):
                    delta = cleaned_text[len(previous_text):]
                    previous_text = cleaned_text

                    yield json.dumps({"type": "content", "text": delta}) + "\n"

        # End of stream
        inference_time = (time.time() - start_time) * 1000

        yield json.dumps({
            "type": "usage",
            "inference_time": inference_time
        }) + "\n"

    except Exception as e:
        print(f"Stream Error: {e}")
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"

# ----------------------------
# Endpoints
# ----------------------------

# Streaming chat endpoint
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    print(f"Received: {request.message}...")
    prompt = format_prompt(request.message)
    request_id = str(uuid.uuid4())
    return StreamingResponse(stream_processor(prompt, request_id), media_type="application/x-ndjson")

# Default chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if llm_engine is None:
        return {"response": "System loading...", "inference_time": 0.0}

    print(f"Received: {request.message}")
    try:
        prompt = format_prompt(request.message)
        request_id = str(uuid.uuid4())
        start_time = time.time()

        results_generator = llm_engine.generate(prompt, sampling_params, request_id)
        output = None
        async for request_output in results_generator:
            output = request_output

        raw_response = output.outputs[0].text

        # Remove the chain-of-thought
        response_text = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

        inference_time = (time.time() - start_time) * 1000 # ms
        print(f"Responding ({inference_time:.2f} ms): {response_text}")
        return {"response": response_text, "inference_time": inference_time}

    except Exception as e:
        print(f"Error: {e}")
        return {"response": f"Error: {str(e)}"}

# Base endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "SFT Model API is running!"}


# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    if public_url:
        print(f"🌐 Public Chat Stream API: {public_url}/chat/stream")
        print(f"🌐 Public Chat API: {public_url}/chat")
        print(f"Health check: {public_url}/")
    else:
        print(f"⚠️ Ngrok not connected or unavailable.")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")



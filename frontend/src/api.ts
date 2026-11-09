import type {
  AnalyzeResponse,
  SynthesizeResponse,
  VoicesResponse,
} from "./types";

/** Thrown for any non-2xx response, carrying the backend error code/message. */
export class ApiRequestError extends Error {
  code: string;
  status: number;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiRequestError(0, "network_error", "Could not reach the server.");
  }

  if (!res.ok) {
    let code = "error";
    let message = `Request failed (${res.status}).`;
    try {
      const data = await res.json();
      if (data?.error) {
        code = data.error.code ?? code;
        message = data.error.message ?? message;
      } else if (data?.detail) {
        // FastAPI validation errors (422) use `detail`.
        code = "validation_error";
        message = Array.isArray(data.detail)
          ? data.detail[0]?.msg ?? message
          : String(data.detail);
      }
    } catch {
      /* non-JSON body; keep defaults */
    }
    throw new ApiRequestError(res.status, code, message);
  }

  return (await res.json()) as T;
}

export const api = {
  getVoices: () => request<VoicesResponse>("/api/voices"),
  analyze: (text: string) => request<AnalyzeResponse>("/api/analyze", { text }),
  synthesize: (text: string, voice: string) =>
    request<SynthesizeResponse>("/api/synthesize", { text, voice }),
};

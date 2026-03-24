/**
 * Server-Sent Events (SSE) endpoint for real-time tool streaming.
 *
 * Clients can subscribe to receive real-time updates during tool execution:
 * - tool_start: Tool execution begins
 * - tool_progress: Intermediate progress updates
 * - tool_end: Tool execution completed
 * - tool_error: Tool execution failed
 *
 * Usage:
 *   GET /api/stream/:sessionId
 *   Accept: text/event-stream
 *
 * Authorization:
 *   - Per-session streams require a valid, existing session (looked up via SessionManager).
 *   - The all-events monitoring endpoint requires an admin token (AG3NT_ADMIN_TOKEN).
 */

import { Router, Request, Response } from "express";
import { EventEmitter } from "events";
import { timingSafeEqual } from "crypto";
import type { SessionManager } from "../session/SessionManager.js";

// Global event bus for tool events
export const toolEventBus = new EventEmitter();
toolEventBus.setMaxListeners(100); // Allow many concurrent SSE connections

// Types for tool events
export interface ToolEvent {
  event_type: "tool_start" | "tool_progress" | "tool_end" | "tool_error";
  session_id: string;
  tool_name: string;
  tool_call_id: string;
  timestamp: number;
  // Additional fields depending on event type
  args?: Record<string, unknown>;
  message?: string;
  progress?: number;
  duration_ms?: number;
  error?: string;
  error_type?: string;
  preview?: string;
  total_size?: number;
}

export interface StreamMessage {
  type: "stream";
  request_id: string;
  event: ToolEvent;
}

/**
 * Forward a stream event from WebSocket to SSE subscribers.
 */
export function forwardStreamEvent(msg: StreamMessage): void {
  const { event } = msg;
  toolEventBus.emit(`session:${event.session_id}`, event);
  toolEventBus.emit("all", event); // For debugging/monitoring
}

/**
 * Perform a timing-safe comparison of two token strings.
 * Returns false if either token is empty or they differ in length/content.
 */
function safeTokenCompare(a: string, b: string): boolean {
  if (!a || !b) return false;
  const bufA = Buffer.from(a, "utf-8");
  const bufB = Buffer.from(b, "utf-8");
  if (bufA.length !== bufB.length) return false;
  return timingSafeEqual(bufA, bufB);
}

/**
 * Create the stream router.
 *
 * @param sessionManager - Used to verify session existence and ownership
 *   for per-session stream subscriptions.
 */
export function createStreamRouter(sessionManager: SessionManager): Router {
  const router = Router();

  /** Admin token for the all-events monitoring endpoint. */
  const adminToken = process.env.AG3NT_ADMIN_TOKEN || "";

  /**
   * SSE endpoint for subscribing to tool events for a session.
   *
   * Authorization: the requested session must exist in the SessionManager.
   * This ensures only sessions created through the authenticated gateway
   * flow can be subscribed to — the session ID acts as a capability token.
   */
  router.get("/:sessionId", (req: Request, res: Response) => {
    const { sessionId } = req.params;

    // --- Authorization: verify the session exists ---
    const session = sessionManager.getSession(sessionId);
    if (!session) {
      res.status(403).json({
        ok: false,
        error: "Forbidden",
        code: "GW-STREAM-001",
        message: "Session not found or access denied",
      });
      return;
    }

    // Set SSE headers
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no"); // Disable nginx buffering
    res.flushHeaders();

    // Send initial connection event
    res.write(`event: connected\ndata: ${JSON.stringify({ session_id: sessionId })}\n\n`);

    // Subscribe to events for this session
    const onEvent = (event: ToolEvent) => {
      try {
        res.write(`event: ${event.event_type}\n`);
        res.write(`data: ${JSON.stringify(event)}\n\n`);
      } catch {
        // Client disconnected
      }
    };

    toolEventBus.on(`session:${sessionId}`, onEvent);

    // Send keepalive every 30 seconds
    const keepaliveInterval = setInterval(() => {
      try {
        res.write(`: keepalive\n\n`);
      } catch {
        // Client disconnected
      }
    }, 30000);

    // Clean up on disconnect
    req.on("close", () => {
      toolEventBus.off(`session:${sessionId}`, onEvent);
      clearInterval(keepaliveInterval);
    });
  });

  /**
   * SSE endpoint for subscribing to all tool events (for monitoring).
   *
   * Authorization: requires a valid admin token via the X-Admin-Token header.
   * The token is compared using crypto.timingSafeEqual to prevent timing attacks.
   */
  router.get("/", (req: Request, res: Response) => {
    // --- Authorization: require admin token ---
    if (!adminToken) {
      // Admin token not configured — endpoint disabled
      res.status(403).json({
        ok: false,
        error: "Forbidden",
        code: "GW-STREAM-002",
        message: "All-events stream is disabled (no admin token configured)",
      });
      return;
    }

    const provided = (req.headers["x-admin-token"] as string) || "";

    if (!safeTokenCompare(provided, adminToken)) {
      res.status(403).json({
        ok: false,
        error: "Forbidden",
        code: "GW-STREAM-003",
        message: "Invalid or missing admin token",
      });
      return;
    }

    // Set SSE headers
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders();

    res.write(`event: connected\ndata: ${JSON.stringify({ type: "all" })}\n\n`);

    const onEvent = (event: ToolEvent) => {
      try {
        res.write(`event: ${event.event_type}\n`);
        res.write(`data: ${JSON.stringify(event)}\n\n`);
      } catch {
        // Client disconnected
      }
    };

    toolEventBus.on("all", onEvent);

    const keepaliveInterval = setInterval(() => {
      try {
        res.write(`: keepalive\n\n`);
      } catch {
        // Client disconnected
      }
    }, 30000);

    req.on("close", () => {
      toolEventBus.off("all", onEvent);
      clearInterval(keepaliveInterval);
    });
  });

  /**
   * Get stream statistics.
   */
  router.get("/stats", (_req: Request, res: Response) => {
    res.json({
      listeners: {
        all: toolEventBus.listenerCount("all"),
      },
    });
  });

  return router;
}

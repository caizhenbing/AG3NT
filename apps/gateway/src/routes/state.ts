/**
 * State API routes for Gateway ↔ Agent state synchronization.
 *
 * These routes allow the Agent Worker to read and update shared session state.
 *
 * Endpoints:
 * - GET /api/state/:sessionId - Get session state
 * - PUT /api/state/:sessionId - Set full session state
 * - PATCH /api/state/:sessionId - Update session state
 * - DELETE /api/state/:sessionId - Delete session state
 * - GET /api/state - List all sessions
 * - GET /api/state/stats - Get store statistics
 */

import { Router, Request, Response } from "express";
import { getStateStore, InMemoryStateStore } from "../state/StateStore.js";
import {
  createSessionState,
  type SessionState,
  type ActivationMode,
  type SessionQuotas,
  type SessionDirective,
  type PendingApproval,
} from "../state/types.js";

const VALID_ACTIVATION_MODES: ActivationMode[] = [
  "always",
  "mention",
  "reply",
  "keyword",
  "off",
];

/**
 * Validate that a value is a non-empty string.
 */
function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

/**
 * Validate that a value is a string or null.
 */
function isStringOrNull(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

/**
 * Validate SessionQuotas structure.
 */
function validateQuotas(quotas: unknown): quotas is SessionQuotas {
  if (typeof quotas !== "object" || quotas === null || Array.isArray(quotas)) {
    return false;
  }
  const q = quotas as Record<string, unknown>;
  return (
    typeof q.maxTokensPerDay === "number" &&
    typeof q.maxRequestsPerHour === "number" &&
    typeof q.tokensUsedToday === "number" &&
    typeof q.requestsThisHour === "number" &&
    typeof q.quotaResetAt === "string"
  );
}

/**
 * Validate a single SessionDirective.
 */
function validateDirective(d: unknown): d is SessionDirective {
  if (typeof d !== "object" || d === null || Array.isArray(d)) {
    return false;
  }
  const dir = d as Record<string, unknown>;
  return (
    typeof dir.id === "string" &&
    typeof dir.content === "string" &&
    typeof dir.priority === "number" &&
    typeof dir.active === "boolean" &&
    typeof dir.createdAt === "string" &&
    (dir.expiresAt === undefined || typeof dir.expiresAt === "string")
  );
}

/**
 * Validate a single PendingApproval.
 */
function validatePendingApproval(p: unknown): p is PendingApproval {
  if (typeof p !== "object" || p === null || Array.isArray(p)) {
    return false;
  }
  const pa = p as Record<string, unknown>;
  return (
    typeof pa.interruptId === "string" &&
    typeof pa.toolName === "string" &&
    typeof pa.args === "object" &&
    pa.args !== null &&
    !Array.isArray(pa.args) &&
    typeof pa.description === "string" &&
    typeof pa.createdAt === "string"
  );
}

/**
 * Validate all fields of a SessionState body for the PUT handler.
 * Returns an error message string if invalid, or null if valid.
 */
function validateSessionStateBody(body: unknown): string | null {
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    return "Request body must be a non-null object";
  }

  const s = body as Record<string, unknown>;

  // Identity fields (required strings)
  if (!isNonEmptyString(s.sessionId)) return "sessionId must be a non-empty string";
  if (!isNonEmptyString(s.channelType)) return "channelType must be a non-empty string";
  if (!isNonEmptyString(s.channelId)) return "channelId must be a non-empty string";
  if (!isNonEmptyString(s.chatId)) return "chatId must be a non-empty string";
  if (!isNonEmptyString(s.userId)) return "userId must be a non-empty string";
  if (s.userName !== undefined && typeof s.userName !== "string") {
    return "userName must be a string if provided";
  }

  // Gateway-managed fields
  if (typeof s.priority !== "number") return "priority must be a number";
  if (!isStringOrNull(s.assignedAgent)) return "assignedAgent must be a string or null";
  if (!Array.isArray(s.directives)) return "directives must be an array";
  for (let i = 0; i < s.directives.length; i++) {
    if (!validateDirective(s.directives[i])) {
      return `directives[${i}] has an invalid structure`;
    }
  }
  if (!validateQuotas(s.quotas)) return "quotas must be a valid SessionQuotas object";
  if (
    typeof s.activationMode !== "string" ||
    !VALID_ACTIVATION_MODES.includes(s.activationMode as ActivationMode)
  ) {
    return `activationMode must be one of: ${VALID_ACTIVATION_MODES.join(", ")}`;
  }
  if (typeof s.paired !== "boolean") return "paired must be a boolean";
  if (s.pairingCode !== undefined && typeof s.pairingCode !== "string") {
    return "pairingCode must be a string if provided";
  }

  // Agent-managed fields
  if (typeof s.messageCount !== "number") return "messageCount must be a number";
  if (!isStringOrNull(s.lastTurnAt)) return "lastTurnAt must be a string or null";
  if (!Array.isArray(s.activeTools)) return "activeTools must be an array";
  for (let i = 0; i < s.activeTools.length; i++) {
    if (typeof s.activeTools[i] !== "string") {
      return `activeTools[${i}] must be a string`;
    }
  }
  if (!Array.isArray(s.pendingApprovals)) return "pendingApprovals must be an array";
  for (let i = 0; i < s.pendingApprovals.length; i++) {
    if (!validatePendingApproval(s.pendingApprovals[i])) {
      return `pendingApprovals[${i}] has an invalid structure`;
    }
  }

  // Timestamps
  if (typeof s.createdAt !== "string") return "createdAt must be a string";
  if (typeof s.updatedAt !== "string") return "updatedAt must be a string";

  // Metadata
  if (typeof s.metadata !== "object" || s.metadata === null || Array.isArray(s.metadata)) {
    return "metadata must be a non-null object";
  }

  // Version
  if (typeof s.version !== "number") return "version must be a number";

  return null;
}

/**
 * Create the state API router.
 */
export function createStateRouter(): Router {
  const router = Router();

  /**
   * Get store statistics.
   */
  router.get("/stats", async (_req: Request, res: Response) => {
    try {
      const store = await getStateStore();

      // Get stats if available (InMemoryStateStore has getStats)
      const stats =
        store instanceof InMemoryStateStore
          ? store.getStats()
          : { sessionCount: (await store.listSessions()).length };

      res.json({
        ok: true,
        stats,
        backend: store instanceof InMemoryStateStore ? "memory" : "redis",
      });
    } catch (err) {
      res.status(500).json({
        ok: false,
        error: err instanceof Error ? err.message : "Unknown error",
      });
    }
  });

  /**
   * List all sessions.
   */
  router.get("/", async (_req: Request, res: Response) => {
    try {
      const store = await getStateStore();
      const sessionIds = await store.listSessions();

      res.json({
        ok: true,
        sessions: sessionIds,
        count: sessionIds.length,
      });
    } catch (err) {
      res.status(500).json({
        ok: false,
        error: err instanceof Error ? err.message : "Unknown error",
      });
    }
  });

  /**
   * Get session state.
   */
  router.get("/:sessionId", async (req: Request, res: Response) => {
    try {
      const { sessionId } = req.params;
      const store = await getStateStore();
      const state = await store.getSession(sessionId);

      if (!state) {
        res.status(404).json({
          ok: false,
          error: "Session not found",
          sessionId,
        });
        return;
      }

      res.json({ ok: true, state });
    } catch (err) {
      res.status(500).json({
        ok: false,
        error: err instanceof Error ? err.message : "Unknown error",
      });
    }
  });

  /**
   * Set full session state.
   */
  router.put("/:sessionId", async (req: Request, res: Response) => {
    try {
      const { sessionId } = req.params;
      const state = req.body;

      // Validate all fields of the session state body
      const validationError = validateSessionStateBody(state);
      if (validationError) {
        res.status(400).json({
          ok: false,
          error: `Invalid state: ${validationError}`,
        });
        return;
      }

      if (state.sessionId !== sessionId) {
        res.status(400).json({
          ok: false,
          error: "Invalid state: sessionId mismatch",
        });
        return;
      }

      const store = await getStateStore();
      await store.setSession(sessionId, state as SessionState);

      res.json({ ok: true, state });
    } catch (err) {
      res.status(500).json({
        ok: false,
        error: err instanceof Error ? err.message : "Unknown error",
      });
    }
  });

  /**
   * Update session state (partial update).
   */
  router.patch("/:sessionId", async (req: Request, res: Response) => {
    try {
      const { sessionId } = req.params;
      const updates = req.body as Partial<SessionState>;
      const source = (req.headers["x-update-source"] as "gateway" | "agent") || "gateway";

      const store = await getStateStore();

      // Check if session exists
      const existing = await store.getSession(sessionId);
      if (!existing) {
        res.status(404).json({
          ok: false,
          error: "Session not found",
          sessionId,
        });
        return;
      }

      const updated = await store.updateSession(sessionId, updates, source);

      res.json({ ok: true, state: updated });
    } catch (err) {
      res.status(500).json({
        ok: false,
        error: err instanceof Error ? err.message : "Unknown error",
      });
    }
  });

  /**
   * Delete session state.
   */
  router.delete("/:sessionId", async (req: Request, res: Response) => {
    try {
      const { sessionId } = req.params;
      const store = await getStateStore();
      const deleted = await store.deleteSession(sessionId);

      if (!deleted) {
        res.status(404).json({
          ok: false,
          error: "Session not found",
          sessionId,
        });
        return;
      }

      res.json({ ok: true, sessionId });
    } catch (err) {
      res.status(500).json({
        ok: false,
        error: err instanceof Error ? err.message : "Unknown error",
      });
    }
  });

  /**
   * Create a new session with defaults.
   */
  router.post("/", async (req: Request, res: Response) => {
    try {
      const { sessionId, channelType, channelId, chatId, userId, userName } =
        req.body;

      if (!sessionId || !channelType || !channelId || !chatId || !userId) {
        res.status(400).json({
          ok: false,
          error: "Missing required fields: sessionId, channelType, channelId, chatId, userId",
        });
        return;
      }

      const store = await getStateStore();

      // Check if session already exists
      const existing = await store.getSession(sessionId);
      if (existing) {
        res.status(409).json({
          ok: false,
          error: "Session already exists",
          sessionId,
        });
        return;
      }

      const state = createSessionState(
        sessionId,
        channelType,
        channelId,
        chatId,
        userId,
        userName
      );

      await store.setSession(sessionId, state);

      res.status(201).json({ ok: true, state });
    } catch (err) {
      res.status(500).json({
        ok: false,
        error: err instanceof Error ? err.message : "Unknown error",
      });
    }
  });

  return router;
}

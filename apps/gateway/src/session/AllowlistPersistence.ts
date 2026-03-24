/**
 * Allowlist Persistence for AG3NT Gateway.
 *
 * Persists the DM pairing allowlist to disk.
 */

import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

export interface AllowlistStore {
  allowlist: string[];
  lastUpdated: string;
}

/**
 * Per-file promise-based write lock.
 *
 * Serializes read-modify-write operations on a given file path so that
 * concurrent callers to addToAllowlist / removeFromAllowlist do not
 * race and silently drop each other's changes.
 */
const fileLocks = new Map<string, Promise<unknown>>();

async function withFileLock<T>(filePath: string, fn: () => Promise<T>): Promise<T> {
  const key = path.resolve(expandPath(filePath));

  // Chain onto whatever is currently pending for this file (or resolve immediately).
  const prev = fileLocks.get(key) ?? Promise.resolve();

  let releaseLock: () => void;
  const gate = new Promise<void>((resolve) => {
    releaseLock = resolve;
  });

  // Register *our* gate so the next caller waits for us.
  fileLocks.set(key, gate);

  // Wait for the previous operation to finish before we start.
  await prev;

  try {
    return await fn();
  } finally {
    // Clean up the map entry if nothing else queued behind us.
    if (fileLocks.get(key) === gate) {
      fileLocks.delete(key);
    }
    releaseLock!();
  }
}

/**
 * Expand ~ to home directory.
 */
function expandPath(filePath: string): string {
  if (filePath.startsWith("~")) {
    return path.join(os.homedir(), filePath.slice(1));
  }
  return filePath;
}

/**
 * Load allowlist from disk.
 */
export async function loadAllowlist(filePath: string): Promise<string[]> {
  const expandedPath = expandPath(filePath);

  try {
    const content = await fs.readFile(expandedPath, "utf-8");
    const store: AllowlistStore = JSON.parse(content);
    return store.allowlist || [];
  } catch (error) {
    // File doesn't exist or is invalid - return empty allowlist
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return [];
    }
    console.warn(
      `[AllowlistPersistence] Failed to load allowlist from ${filePath}:`,
      error
    );
    return [];
  }
}

/**
 * Save allowlist to disk.
 */
export async function saveAllowlist(
  filePath: string,
  allowlist: string[]
): Promise<void> {
  const expandedPath = expandPath(filePath);

  // Ensure directory exists
  const dir = path.dirname(expandedPath);
  await fs.mkdir(dir, { recursive: true });

  const store: AllowlistStore = {
    allowlist,
    lastUpdated: new Date().toISOString(),
  };

  await fs.writeFile(expandedPath, JSON.stringify(store, null, 2), "utf-8");
}

/**
 * Add an entry to the allowlist and persist.
 */
export async function addToAllowlist(
  filePath: string,
  entry: string
): Promise<string[]> {
  return withFileLock(filePath, async () => {
    const allowlist = await loadAllowlist(filePath);
    if (!allowlist.includes(entry)) {
      allowlist.push(entry);
      await saveAllowlist(filePath, allowlist);
    }
    return allowlist;
  });
}

/**
 * Remove an entry from the allowlist and persist.
 */
export async function removeFromAllowlist(
  filePath: string,
  entry: string
): Promise<string[]> {
  return withFileLock(filePath, async () => {
    let allowlist = await loadAllowlist(filePath);
    allowlist = allowlist.filter((e) => e !== entry);
    await saveAllowlist(filePath, allowlist);
    return allowlist;
  });
}


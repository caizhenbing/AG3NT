/**
 * Channel Registry for AG3NT Gateway.
 *
 * Manages registration and lifecycle of channel adapters.
 */

import type {
  IChannelAdapter,
  MessageHandler,
  ChannelMessage,
  ChannelResponse,
} from "./types.js";

/**
 * Event emitted when a channel adapter state changes.
 */
export type ChannelEvent =
  | { type: "connected"; adapterId: string }
  | { type: "disconnected"; adapterId: string }
  | { type: "error"; adapterId: string; error: Error }
  | { type: "message"; adapterId: string; message: ChannelMessage };

export type ChannelEventHandler = (event: ChannelEvent) => void;

/**
 * Manages all channel adapters in the Gateway.
 */
export class ChannelRegistry {
  private adapters: Map<string, IChannelAdapter> = new Map();
  private eventHandlers: ChannelEventHandler[] = [];
  private globalMessageHandler: MessageHandler | null = null;

  /**
   * Register a channel adapter.
   * @param adapter - The adapter to register
   */
  register(adapter: IChannelAdapter): void {
    if (this.adapters.has(adapter.id)) {
      throw new Error(`Adapter already registered: ${adapter.id}`);
    }

    // Wire up the message handler
    adapter.onMessage(async (message) => {
      // Emit event
      this.emit({ type: "message", adapterId: adapter.id, message });

      // Call global handler if set
      if (this.globalMessageHandler) {
        return this.globalMessageHandler(message);
      }
      return undefined;
    });

    this.adapters.set(adapter.id, adapter);
  }

  /**
   * Unregister a channel adapter.
   * Will disconnect if currently connected.
   * @param adapterId - The adapter ID to unregister
   */
  async unregister(adapterId: string): Promise<void> {
    const adapter = this.adapters.get(adapterId);
    if (!adapter) return;

    if (adapter.isConnected()) {
      await adapter.disconnect();
    }

    this.adapters.delete(adapterId);
  }

  /**
   * Get an adapter by ID.
   */
  get(adapterId: string): IChannelAdapter | undefined {
    return this.adapters.get(adapterId);
  }

  /**
   * Get all adapters of a specific type.
   */
  getByType(type: string): IChannelAdapter[] {
    return Array.from(this.adapters.values()).filter((a) => a.type === type);
  }

  /**
   * Get all registered adapters.
   */
  all(): IChannelAdapter[] {
    return Array.from(this.adapters.values());
  }

  /**
   * Connect all registered adapters.
   * Uses Promise.allSettled so a single adapter failure does not prevent
   * other adapters from connecting. If all adapters fail, throws an
   * aggregate error. If only some fail, logs warnings and continues.
   */
  async connectAll(): Promise<void> {
    const adapterList = Array.from(this.adapters.values());
    if (adapterList.length === 0) return;

    const results = await Promise.allSettled(
      adapterList.map(async (adapter) => {
        await adapter.connect();
        this.emit({ type: "connected", adapterId: adapter.id });
        return adapter.id;
      })
    );

    const errors: { adapterId: string; error: Error }[] = [];

    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      if (result.status === "rejected") {
        const adapter = adapterList[i];
        const error =
          result.reason instanceof Error
            ? result.reason
            : new Error(String(result.reason));
        errors.push({ adapterId: adapter.id, error });
        this.emit({ type: "error", adapterId: adapter.id, error });
      }
    }

    if (errors.length > 0 && errors.length === adapterList.length) {
      throw new Error(
        `All channel adapters failed to connect: ${errors.map((e) => `${e.adapterId} (${e.error.message})`).join(", ")}`
      );
    }

    if (errors.length > 0) {
      for (const { adapterId, error } of errors) {
        console.warn(
          `[ChannelRegistry] Adapter "${adapterId}" failed to connect: ${error.message}`
        );
      }
    }
  }

  /**
   * Disconnect all registered adapters.
   */
  async disconnectAll(): Promise<void> {
    const disconnectPromises = Array.from(this.adapters.values()).map(
      async (adapter) => {
        try {
          await adapter.disconnect();
          this.emit({ type: "disconnected", adapterId: adapter.id });
        } catch (error) {
          this.emit({
            type: "error",
            adapterId: adapter.id,
            error: error instanceof Error ? error : new Error(String(error)),
          });
        }
      }
    );

    await Promise.all(disconnectPromises);
  }

  /**
   * Set the global message handler for all adapters.
   * This is called when any adapter receives a message.
   */
  setMessageHandler(handler: MessageHandler): void {
    this.globalMessageHandler = handler;
  }

  /**
   * Register an event handler for channel events.
   */
  onEvent(handler: ChannelEventHandler): void {
    this.eventHandlers.push(handler);
  }

  private emit(event: ChannelEvent): void {
    for (const handler of this.eventHandlers) {
      try {
        handler(event);
      } catch {
        // Ignore handler errors
      }
    }
  }
}


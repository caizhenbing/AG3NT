// 临时完整日志模块（含 log 方法），供编译使用
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: LogLevel;
  source: string;
  message: string;
  data?: any;
}

export class GatewayLogs {
  private entries: LogEntry[] = [];
  private subscribers: Array<(entry: LogEntry) => void> = [];
  private fileLoggingEnabled = false;
  private minLevel: LogLevel = 'info';
  private logDir?: string;

  enableFileLogging(opts: { dir: string; minLevel: LogLevel }) {
    this.fileLoggingEnabled = true;
    this.logDir = opts.dir;
    this.minLevel = opts.minLevel;
  }

  private addEntry(level: LogLevel, source: string, message: string, data?: any): LogEntry {
    const entry: LogEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      timestamp: new Date(),
      level,
      source,
      message,
      data,
    };
    this.entries.push(entry);
    this.subscribers.forEach(cb => cb(entry));
    return entry;
  }

  // 通用 log 方法（同时支持 2 参数和 4 参数调用）
  log(level: LogLevel, source: string, message: string, data?: any): void;
  log(source: string, message: string): void;
  log(arg1: string, arg2: string, arg3?: string | any, arg4?: any): void {
    if (arguments.length === 2) {
      this.info(arg1, arg2);
    } else {
      const level = arg1 as LogLevel;
      const source = arg2;
      const message = typeof arg3 === 'string' ? arg3 : '';
      const data = arguments.length === 4 ? arg4 : arg3;
      this.addEntry(level, source, message, data);
    }
  }

  info(source: string, message: string, data?: any, _extra?: any): void {
    this.addEntry('info', source, message, data);
  }

  warn(source: string, message: string, data?: any): void {
    this.addEntry('warn', source, message, data);
  }

  error(source: string, message: string, data?: any): void {
    this.addEntry('error', source, message, data);
  }

  debug(source: string, message: string, data?: any): void {
    this.addEntry('debug', source, message, data);
  }

  subscribe(callback: (entry: LogEntry) => void) {
    this.subscribers.push(callback);
  }

  getRecent(count: number, level?: LogLevel): LogEntry[] {
    let filtered = this.entries;
    if (level) {
      filtered = filtered.filter(e => e.level === level);
    }
    return filtered.slice(-count).reverse();
  }

  clear() {
    this.entries = [];
  }
}

export const gatewayLogs = new GatewayLogs();

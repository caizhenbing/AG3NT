import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import YAML from "yaml";
import { ConfigSchema, type Config } from "./schema.js";

function expandHome(p: string): string {
  if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2));
  return p;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function deepMerge(
  base: Record<string, unknown>,
  override: Record<string, unknown>,
): Record<string, unknown> {
  const result: Record<string, unknown> = { ...base };
  for (const key of Object.keys(override)) {
    const baseVal = base[key];
    const overrideVal = override[key];
    if (isPlainObject(baseVal) && isPlainObject(overrideVal)) {
      result[key] = deepMerge(baseVal, overrideVal);
    } else {
      result[key] = overrideVal;
    }
  }
  return result;
}

async function readYamlIfExists(filePath: string): Promise<Record<string, unknown>> {
  try {
    const text = await fs.readFile(filePath, "utf8");
    return YAML.parse(text) ?? {};
  } catch {
    return {};
  }
}

export async function loadConfig(): Promise<Config> {
  const defaultPath = path.resolve(process.cwd(), "../../config/default-config.yaml");
  const userPath = expandHome("~/.ag3nt/config.yaml");

  const base = await readYamlIfExists(defaultPath);
  const user = await readYamlIfExists(userPath);

  const merged = deepMerge(base, user);
  const parsed = ConfigSchema.safeParse(merged);
  if (!parsed.success) {
    const message = parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("\n");
    throw new Error(`Invalid config\n${message}`);
  }
  return parsed.data;
}

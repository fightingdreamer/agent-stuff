/**
 * /delegate command and delegate tool
 *
 * Forks a task to an in-process sub-agent with full session context.
 * Renders as a custom tool (Box with border/bg via ToolExecutionComponent).
 */

import {
	convertToLlm,
	codingTools,
	createAgentSession,
	createExtensionRuntime,
	serializeConversation,
	truncateTail,
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	SessionManager,
	type AgentSession,
	type ExtensionAPI,
	type ExtensionContext,
	type ResourceLoader,
} from "@mariozechner/pi-coding-agent";
import { type AssistantMessage } from "@mariozechner/pi-ai";
import { Text } from "@mariozechner/pi-tui";
import { Type } from "typebox";

// ── resource loader ────────────────────────────────────────────────────────

function createDelegateResourceLoader(
	ctx: ExtensionContext,
): ResourceLoader {
	const extensionsResult = { extensions: [], errors: [], runtime: createExtensionRuntime() };
	return {
		getExtensions: () => extensionsResult,
		getSkills: () => ({ skills: [], diagnostics: [] }),
		getPrompts: () => ({ prompts: [], diagnostics: [] }),
		getThemes: () => ({ themes: [], diagnostics: [] }),
		getAgentsFiles: () => ({ agentsFiles: [] }),
		getSystemPrompt: () => ctx.getSystemPrompt(),
		getAppendSystemPrompt: () => [],
		extendResources: () => {},
		reload: async () => {},
	};
}

// ── session context ────────────────────────────────────────────────────────

function serializeSessionContext(ctx: ExtensionContext): string {
	try {
		const entries = ctx.sessionManager.getBranch();
		// Skip last 2 entries (assistant tool call + user delegate message)
		// so the sub-agent only sees the conversation before the delegation
		const filtered = entries.slice(0, -2);
		const messages = filtered
			.map((e) => {
				if (e.type === "message") return e.message;
				if (e.type === "compaction") return {
					role: "compactionSummary" as const,
					summary: e.summary,
					tokensBefore: e.tokensBefore,
					timestamp: new Date(e.timestamp).getTime(),
				};
				if (e.type === "branch_summary") return {
					role: "branchSummary" as const,
					summary: e.summary,
					fromId: e.fromId,
					timestamp: new Date(e.timestamp).getTime(),
				};
				return undefined;
			})
			.filter(Boolean) as any[];
		return serializeConversation(convertToLlm(messages));
	} catch {
		return "(could not serialize context)";
	}
}

// ── sub-agent helpers ──────────────────────────────────────────────────────

function lastAssistantText(session: AgentSession): string {
	for (let i = session.state.messages.length - 1; i >= 0; i--) {
		const m = session.state.messages[i];
		if (m.role === "assistant") {
			const msg = m as AssistantMessage;
			return msg.content.filter((c) => c.type === "text").map((c) => c.text).join("\n").trim();
		}
	}
	return "(no response)";
}

function lastStopReason(session: AgentSession): string | undefined {
	const last = session.state.messages[session.state.messages.length - 1] as AssistantMessage | undefined;
	return last?.stopReason;
}

function lastErrorMessage(session: AgentSession): string | undefined {
	const last = session.state.messages[session.state.messages.length - 1] as AssistantMessage | undefined;
	return last?.errorMessage;
}

// ── sub-agent execution ────────────────────────────────────────────────────

async function runSubAgent(
	task: string,
	ctx: ExtensionContext,
	pi: ExtensionAPI,
	onOutput?: (text: string) => void,
	signal?: AbortSignal,
): Promise<{ result: string; error?: string }> {
	if (!ctx.model) return { result: "", error: "No active model selected." };
	const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
	if (!auth.ok) return { result: "", error: auth.error };

	const { session } = await createAgentSession({
		sessionManager: SessionManager.inMemory(),
		model: ctx.model,
		modelRegistry: ctx.modelRegistry as AgentSession["modelRegistry"],
		thinkingLevel: pi.getThinkingLevel(),
		tools: codingTools,
		resourceLoader: createDelegateResourceLoader(ctx),
	});

	const contextSummary = serializeSessionContext(ctx);
	const promptText = contextSummary
		? `${contextSummary}\n\nAbove is the conversation history, ${task}`
		: task;

	let accumulated = "";
	let rawLines: string[] = [];
	const MAX_PREVIEW_LINES = 4;
	const flushPreview = () => {
		if (!onOutput) return;
		onOutput(rawLines.slice(-MAX_PREVIEW_LINES).join("\n"));
	};
	const updateFromText = (text: string) => {
		if (!text) return;
		rawLines = text.split("\n");
		flushPreview();
	};
	const unsubscribe = onOutput
		? session.subscribe((event: any) => {
				if (event.type === "message_start" || event.type === "message_update") {
					const m = event.message as { role?: string; content?: Array<{ type: string; text?: string }> };
					if (m.role === "assistant" && m.content) {
						const text = m.content.filter((c) => c.type === "text").map((c) => c.text).join("\n").trim();
						if (text && text !== accumulated) {
							accumulated = text;
							updateFromText(text);
						}
					}
				}
				if (event.type === "tool_execution_start") {
					const e = event as { toolName?: string; args?: Record<string, unknown> };
					const arg = e.args ? (typeof e.args.command === "string" ? e.args.command : typeof e.args.path === "string" ? e.args.path : "") : "";
					rawLines.push(`[${e.toolName ?? "?"}${arg ? `: ${arg}` : ""}]`);
					flushPreview();
				}
				if (event.type === "tool_execution_update") {
					const e = event as { partialResult?: { content?: Array<{ text?: string }> } };
					const text = e.partialResult?.content?.map((c) => c.text ?? "").join("") ?? "";
					updateFromText(text);
				}
				if (event.type === "tool_execution_end") {
					rawLines.push("");
					flushPreview();
				}
		  })
		: undefined;

	// Handle abort signal (user cancel, Esc, etc.)
	if (signal) {
		signal.addEventListener("abort", () => {
			try { session.abort(); } catch { /* ignore */ }
		}, { once: true });
	}

	try {
		await session.prompt(promptText, { source: "extension" });
		unsubscribe?.();
		const reason = lastStopReason(session);
		if (reason === "aborted") return { result: "", error: "Sub-agent request aborted." };
		if (reason === "error") return { result: "", error: lastErrorMessage(session) || "Sub-agent request failed." };
		return { result: lastAssistantText(session) };
	} catch (error: any) {
		unsubscribe?.();
		return { result: "", error: error.message || "Sub-agent failed." };
	} finally {
		try { await session.abort(); } catch { /* ignore */ }
		session.dispose();
	}
}

// ── result formatting ──────────────────────────────────────────────────────

function formatToolResult(text: string): string {
	const truncation = truncateTail(text, { maxLines: DEFAULT_MAX_LINES, maxBytes: DEFAULT_MAX_BYTES });
	let result = truncation.content;
	if (truncation.truncated) {
		result += `\n\n[Output truncated: ${truncation.outputLines} of ${truncation.totalLines} lines (showing last portion). Full output in tool details.]`;
	}
	return result;
}

// ── extension ──────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI): void {
	pi.registerCommand("delegate", {
		description:
			"Run a sub-agent with the assigned task and full conversation context. " +
			"Reports back with its result. " +
			"Usage: /delegate <task description>",
		handler: async (args, ctx) => {
			const task = args.trim();
			if (!task) {
				ctx.ui.notify("Usage: /delegate <task description>", "warning");
				return;
			}

			ctx.ui.setStatus("delegate", `Delegating: ${task.slice(0, 40)}...`);
			const result = await runSubAgent(task, ctx, pi);
			ctx.ui.setStatus("delegate", undefined);

			if (result.error) {
				ctx.ui.notify(`Delegate failed: ${result.error}`, "error");
				return;
			}

			pi.sendUserMessage(`**Delegate result** (task: ${task}):\n\n${result.result}`, { deliverAs: "nextTurn" });
		},
	});

	pi.registerTool({
		name: "delegate",
		label: "Delegate",
		description:
			"Run a sub-agent with the assigned task and full conversation context. " +
			"The sub-agent has access to all tools (read, bash, edit, write) and reports back with its result. " +
			"Can speed up running independent tasks in parallel.",
		parameters: Type.Object({
			task: Type.String({ description: "What the sub-agent should do." }),
		}),
		renderCall(args, _theme) {
			return new Text(_theme.fg("toolTitle", _theme.bold(`delegate: ${(args as any).task}`)), 0, 0);
		},
		renderResult(result, _options, _theme) {
			const text = result.content
				.filter((c: any) => c.type === "text")
				.map((c: any) => c.text)
				.join("\n");
			return new Text(`\n${text}`, 0, 0);
		},
		async execute(_toolCallId, params, signal, onUpdate, ctx) {
			const result = await runSubAgent(params.task, ctx, pi, (text) => {
				onUpdate?.({
					content: [{ type: "text", text }],
				});
			}, signal);

			if (result.error) {
				return {
					content: [{ type: "text", text: `Delegate failed: ${result.error}` }],
					details: { error: result.error },
				};
			}

			return {
				content: [{ type: "text", text: formatToolResult(result.result) }],
				details: { result: result.result },
			};
		},
	});
}

"use client";

const TOKEN = /("(?:\\.|[^\\"])*"(\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?)/g;

function highlight(json: string): string {
  const esc = json.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  return esc.replace(TOKEN, (m) => {
    let cls = "text-amber-300";
    if (m.startsWith('"')) cls = m.endsWith(":") ? "text-cyan-300" : "text-emerald-300";
    else if (/^(true|false|null)$/.test(m)) cls = "text-violet-300";
    return `<span class="${cls}">${m}</span>`;
  });
}

/** macOS-style window rendering syntax-highlighted JSON (keys and numbers in mono). */
export default function TerminalWindow({ title, data }: { title: string; data: unknown }) {
  const html = highlight(JSON.stringify(data, null, 2));
  return (
    <div className="overflow-hidden rounded-lg border border-white/[0.08] bg-black/70 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.08)]">
      <div className="flex items-center gap-2 border-b border-white/[0.08] bg-white/[0.02] px-3 py-2">
        <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]" />
        <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]" />
        <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]" />
        <span className="ml-3 truncate font-mono text-[11px] text-zinc-500">{title}</span>
      </div>
      <pre
        className="max-h-72 overflow-auto p-4 font-mono text-[12px] leading-relaxed text-zinc-300"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}

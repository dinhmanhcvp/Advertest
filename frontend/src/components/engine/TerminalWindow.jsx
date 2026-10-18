"use client";

const TOKEN = /("(?:\\.|[^\\"])*"(\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?)/g;

function highlight(json) {
  const esc = json.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  return esc.replace(TOKEN, (m) => {
    let cls = "text-amber-300";
    if (m.startsWith('"')) cls = m.endsWith(":") ? "text-cyan-300" : "text-emerald-300";
    else if (/^(true|false|null)$/.test(m)) cls = "text-violet-300";
    return `<span class="${cls}">${m}</span>`;
  });
}

export default function TerminalWindow({ title = "audit_trail.json", data }) {
  const html = highlight(JSON.stringify(data ?? {}, null, 2));
  return (
    <div className="overflow-hidden rounded-lg border border-white/10 bg-black/60 backdrop-blur-xl">
      <div className="flex items-center gap-2 border-b border-white/10 px-3 py-2">
        <span className="h-2.5 w-2.5 rounded-full bg-red-500/80" />
        <span className="h-2.5 w-2.5 rounded-full bg-amber-400/80" />
        <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/80" />
        <span className="ml-3 font-mono text-[11px] text-zinc-500">{title}</span>
      </div>
      <pre
        className="max-h-72 overflow-auto p-4 font-mono text-[12px] leading-relaxed text-zinc-300"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}

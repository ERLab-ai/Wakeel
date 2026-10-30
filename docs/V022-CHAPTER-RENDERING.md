# Authored full chapter rendering — v0.22

**Scope:** user-interface conversion of the *existing, already signed-off* chapter Markdown text. No scientific content, simulator, or frozen result changes.

### Security and reproducibility

- Read **exactly 18 files**, \`content/chapter-01/{ar,en}.md\` through chapter 09. Paths are fixed by code and are never determined by user input.
- Every original byte sequence must match the per-file digest in \`baseline/v0.21.3/PUBLIC-MANIFEST.json\`. The static build fails closed on any mismatch.
- Markdown is rendered at build time with \`markdown-it-py==4.2.0\`, CommonMark, HTML disabled, links validated by renderer, and GitHub-style pipe tables enabled. Rendered HTML is derived from **trusted, approved** Markdown only.
- Do not insert untrusted user input or registry labels as HTML; those stay \`textContent\` DOM nodes.
- The browser reads the already generated HTML from the trusted six-file \`dist/data/learning.json\`; the archive and review data never enter \`dist\`.
- Every rendered article exposes its own source path, source SHA-256 and byte count in the generated catalog, and G7 tests require both languages.
- Each chapter can be expanded within a station and stays accessible from its first station; opening a chapter does not force prior station completion.

### Technical limitations

Scientific expressions encoded as fenced blocks and inline code are preserved in their source typography. Markdown raw HTML is deliberately escaped. No new diagram rendering, MathJax, or user-generated Markdown is supported in this gate. Keyboard/mobile and independent human linguistic/scientific review are still separate release gates.

### Run

\`\`\`sh
python -m pip install markdown-it-py==4.2.0
npm run check
npm run serve
\`\`\`

**Source immutability:** Compare Git trees under \`baseline/v0.21.3/\` before merge; they must remain identical to P0.

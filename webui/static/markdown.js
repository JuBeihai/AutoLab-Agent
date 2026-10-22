(function (root) {
  function escape(text) {
    return String(text).replace(/[&<>"']/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[character]);
  }

  function inline(text, depth = 0) {
    if (depth > 8) return escape(text);
    const pattern = /`([^`\n]+)`|\*\*([^\n]+?)\*\*|__([^\n]+?)__|\*([^*\n]+)\*|\[([^\]\n]+)\]\(([^\s)]+)\)/g;
    let result = '';
    let cursor = 0;
    for (const match of text.matchAll(pattern)) {
      result += escape(text.slice(cursor, match.index));
      if (match[1] !== undefined) result += `<code>${escape(match[1])}</code>`;
      else if (match[2] !== undefined || match[3] !== undefined) {
        result += `<strong>${inline(match[2] ?? match[3], depth + 1)}</strong>`;
      } else if (match[4] !== undefined) result += `<em>${inline(match[4], depth + 1)}</em>`;
      else {
        const label = escape(match[5]);
        const address = match[6];
        result += (/^https?:\/\/[^\s]+$/i.test(address) || /^\/api\/code-workspace\/[A-Za-z0-9_-]+\/download\?path=[A-Za-z0-9_.%~-]+$/.test(address))
          ? `<a href="${escape(address)}" target="_blank" rel="noopener noreferrer">${label}</a>`
          : escape(match[0]);
      }
      cursor = match.index + match[0].length;
    }
    return result + escape(text.slice(cursor));
  }

  function tableCells(line) {
    let value = line.trim();
    if (value.startsWith('|')) value = value.slice(1);
    if (/(?<!\\)\|$/.test(value)) value = value.slice(0, -1);
    const cells = [];
    let cell = '';
    let fence = 0;
    for (let index = 0; index < value.length; index++) {
      const character = value[index];
      if (character === '\\' && value[index + 1] === '|') {
        cell += '|';
        index++;
      } else if (character === '`') {
        let end = index;
        while (value[end] === '`') end++;
        const length = end - index;
        if (!fence) fence = length;
        else if (fence === length) fence = 0;
        cell += value.slice(index, end);
        index = end - 1;
      } else if (character === '|' && !fence) {
        cells.push(cell.trim());
        cell = '';
      } else cell += character;
    }
    cells.push(cell.trim());
    return cells;
  }

  function render(text) {
    const lines = String(text).replace(/\r\n?/g, '\n').split('\n');
    const blocks = [];
    const paragraph = [];
    let list = null;
    let fence = null;
    let code = [];

    function flushParagraph() {
      if (paragraph.length) blocks.push(`<p>${paragraph.map(line => inline(line)).join('<br>')}</p>`);
      paragraph.length = 0;
    }

    function flushList() {
      if (!list) return;
      const start = list.tag === 'ol' ? ` start="${list.start}"` : '';
      blocks.push(`<${list.tag}${start}>${list.items.map(item => `<li>${item.map(line => inline(line)).join('<br>')}</li>`).join('')}</${list.tag}>`);
      list = null;
    }

    for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
      const line = lines[lineIndex];
      if (fence) {
        if (new RegExp(`^\\s{0,3}${fence.character}{${fence.length},}\\s*$`).test(line)) {
          blocks.push(`<pre><code>${escape(code.join('\n'))}</code></pre>`);
          fence = null;
          code = [];
        } else code.push(line);
        continue;
      }
      const opening = line.match(/^\s{0,3}(`{3,}|~{3,})(.*)$/);
      if (opening) {
        flushParagraph();
        flushList();
        fence = { character: opening[1][0], length: opening[1].length };
        continue;
      }
      if (!line.trim()) {
        flushParagraph();
        flushList();
        continue;
      }
      const headers = tableCells(line);
      const separators = tableCells(lines[lineIndex + 1] || '');
      if (line.includes('|') && headers.length === separators.length &&
          separators.every(cell => /^:?-{3,}:?$/.test(cell))) {
        flushParagraph();
        flushList();
        const alignments = separators.map(cell => cell.startsWith(':') && cell.endsWith(':')
          ? 'center' : cell.endsWith(':') ? 'right' : 'left');
        const row = (cells, tag) => '<tr>' + headers.map((header, column) =>
          `<${tag} class="md-align-${alignments[column]}"${tag === 'th' ? ' scope="col"' : ''}>${inline(cells[column] || '')}</${tag}>`
        ).join('') + '</tr>';
        const rows = [];
        lineIndex++;
        while (lineIndex + 1 < lines.length && lines[lineIndex + 1].trim() &&
               lines[lineIndex + 1].includes('|') && !/^\s*(`{3,}|~{3,}|#{1,6}\s|>)/.test(lines[lineIndex + 1])) {
          const cells = tableCells(lines[++lineIndex]);
          if (cells.length > headers.length) {
            cells[headers.length - 1] = cells.slice(headers.length - 1).join(' | ');
          }
          rows.push(row(cells, 'td'));
        }
        blocks.push(`<div class="markdown-table-wrap" role="region" aria-label="回复中的数据表格" tabindex="0"><table><thead>${row(headers, 'th')}</thead><tbody>${rows.join('')}</tbody></table></div>`);
        continue;
      }
      const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+)$/);
      const item = line.match(/^\s*(?:([-+*])|([0-9]{1,9})[.)])\s+(.+)$/);
      const quote = line.match(/^\s{0,3}>\s?(.*)$/);
      if (heading || quote || /^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        flushParagraph();
        flushList();
        if (heading) blocks.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
        else if (quote) blocks.push(`<blockquote>${inline(quote[1])}</blockquote>`);
        else blocks.push('<hr>');
      } else if (item) {
        flushParagraph();
        const tag = item[2] ? 'ol' : 'ul';
        if (list && list.tag !== tag) flushList();
        if (!list) list = { tag, start: Number(item[2] || 1), items: [] };
        list.items.push([item[3]]);
      } else if (list && /^\s{2,}\S/.test(line)) {
        list.items[list.items.length - 1].push(line.trim());
      } else {
        flushList();
        paragraph.push(line);
      }
    }
    if (fence) blocks.push(`<pre><code>${escape(code.join('\n'))}</code></pre>`);
    flushParagraph();
    flushList();
    return blocks.join('\n');
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { render };
  else root.ETSMarkdown = { render };
})(globalThis);

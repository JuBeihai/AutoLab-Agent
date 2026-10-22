(function (root) {
  function detectRunIntent(text) {
    const normalized = text.trim().replace(/\s+/g, "");
    if (/[?？]|为什么|怎么|如何|能否|是否|可以|不要|别|不想|取消|停止|失败|如果|假如|\b(?:not|why|how|stop|don't)\b/i.test(text)) return null;
    const prefix = normalized.match(/^(?:请)?(?:只跑baseline|仅跑baseline|启动(?:实验|训练)?|开始(?:实验|训练)|运行(?:实验|训练)?|执行(?:实验|训练)?|开跑|跑|run|start|launch)/i);
    if (!prefix) return null;
    let remaining = normalized.slice(prefix[0].length).replace(/[,，、;；。.!！]/g, "");
    const roundsMatch = remaining.match(/(?:跑|运行)?(\d+)(?:轮|rounds?)/i) || remaining.match(/rounds?[=:：]?(\d+)/i);
    const proposalsMatch = remaining.match(/(?:每轮)?(\d+)个候选/) || remaining.match(/候选数[=:：为]?(\d+)/) || remaining.match(/max[_-]?proposals?[=:：]?(\d+)/i);
    if (roundsMatch) remaining = remaining.replace(roundsMatch[0], "");
    if (proposalsMatch) remaining = remaining.replace(proposalsMatch[0], "");
    const real = /真实数据|real/i.test(remaining);
    const mock = /合成数据|mock/i.test(remaining);
    if (real && mock) return null;
    const noAgent = /baseline/i.test(normalized);
    remaining = remaining.replace(/(?:使用|用)?(?:真实数据|合成数据|mock数据|mock|real)/gi, "")
      .replace(/(?:只跑|仅跑|only)?baseline/gi, "");
    if (remaining) return null;
    return {
      rounds: roundsMatch ? Number(roundsMatch[1]) : null,
      maxProposals: proposalsMatch ? Number(proposalsMatch[1]) : null,
      mock: real ? false : mock ? true : null,
      noAgent: noAgent ? true : null,
    };
  }

  function buildRunConfig(intent, current) {
    return {
      ...current,
      rounds: intent.rounds ?? current.rounds,
      max_proposals: intent.maxProposals ?? current.max_proposals,
      mock: intent.mock ?? current.mock,
      no_agent: intent.noAgent ?? current.no_agent,
    };
  }

  const controls = { detectRunIntent, buildRunConfig };
  if (typeof module !== "undefined" && module.exports) module.exports = controls;
  else root.AutoLabControls = controls;
})(globalThis);

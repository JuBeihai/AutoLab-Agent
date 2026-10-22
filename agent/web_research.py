"""Standalone public research, usable without Spark, model credentials or LLM access."""
import argparse
import json
from pathlib import Path
from skills.web_research import TOPICS, collect


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', choices=TOPICS, default='regularization')
    parser.add_argument('--refresh', action='store_true', help='忽略24小时缓存，重新访问公开资料')
    parser.add_argument('--query', help='公开搜索关键词，不要包含业务数据或凭证')
    parser.add_argument('--url', action='append', dest='urls', help='读取指定 HTTPS 网页或 PDF，可重复最多5次')
    parser.add_argument('--abstract-only', action='store_true', help='仅收集摘要与网页片段，不下载论文全文')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    report = collect(root, args.topic, refresh=args.refresh, query=args.query, urls=args.urls,
                     read_full_text=False if args.abstract_only else None)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['sources'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

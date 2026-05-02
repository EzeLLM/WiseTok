#!/usr/bin/env python3
"""
CLI tool for log parsing and deduplication with argparse.
Includes progress bars, color output, and config file loading.
"""
import argparse
import sys
import json
import configparser
from pathlib import Path
from typing import List, Set, Dict, Any
import hashlib
from datetime import datetime
import subprocess


class Colors:
    '''ANSI color codes for terminal output.'''
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'


def print_progress(current: int, total: int, prefix: str = '') -> None:
    '''Print a simple progress indicator.'''
    if total == 0:
        return
    percent = 100 * (current / float(total))
    filled = int(20 * current / float(total))
    bar = '█' * filled + '░' * (20 - filled)
    status = f'{Colors.BLUE}{prefix}{bar}{Colors.RESET} {percent:.1f}% ({current}/{total})'
    sys.stdout.write(f'\r{status}')
    sys.stdout.flush()


def load_config(config_path: Path) -> Dict[str, Any]:
    '''Load configuration from INI file.'''
    config = configparser.ConfigParser()
    if config_path.exists():
        config.read(config_path)
        return {
            'log_pattern': config.get('parsing', 'pattern', fallback=r'\[(.*?)\]'),
            'hash_algorithm': config.get('dedup', 'algorithm', fallback='md5'),
            'ignore_case': config.getboolean('parsing', 'ignore_case', fallback=False),
            'output_format': config.get('output', 'format', fallback='json'),
        }
    return {
        'log_pattern': r'\[(.*?)\]',
        'hash_algorithm': 'md5',
        'ignore_case': False,
        'output_format': 'json',
    }


def parse_logs(log_file: Path, pattern: str, ignore_case: bool) -> List[Dict[str, str]]:
    '''Parse log file and extract entries matching pattern.'''
    import re
    flags = re.IGNORECASE if ignore_case else 0
    regex = re.compile(pattern, flags)

    entries = []
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                matches = regex.findall(line)
                if matches:
                    for match in matches:
                        entries.append({
                            'line_num': line_num,
                            'content': match,
                            'timestamp': datetime.now().isoformat(),
                        })
    except IOError as e:
        print(f'{Colors.RED}Error reading {log_file}: {e}{Colors.RESET}', file=sys.stderr)

    return entries


def deduplicate_entries(entries: List[Dict[str, str]], algorithm: str) -> Dict[str, List[Dict]]:
    '''Group entries by hash to identify duplicates.'''
    hash_map: Dict[str, List[Dict]] = {}

    for entry in entries:
        content = entry['content']
        if algorithm == 'md5':
            h = hashlib.md5(content.encode()).hexdigest()
        elif algorithm == 'sha256':
            h = hashlib.sha256(content.encode()).hexdigest()
        else:
            h = hashlib.md5(content.encode()).hexdigest()

        if h not in hash_map:
            hash_map[h] = []
        hash_map[h].append(entry)

    return hash_map


def export_json(data: Dict[str, Any], output_file: Path) -> None:
    '''Export results as JSON.'''
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'{Colors.GREEN}✓ Exported to {output_file}{Colors.RESET}')


def export_csv(data: Dict[str, Any], output_file: Path) -> None:
    '''Export results as CSV.'''
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hash', 'count', 'sample_content', 'line_numbers'])

        for h, entries in data.items():
            lines = ','.join(str(e['line_num']) for e in entries)
            writer.writerow([h, len(entries), entries[0]['content'], lines])

    print(f'{Colors.GREEN}✓ Exported to {output_file}{Colors.RESET}')


def main():
    parser = argparse.ArgumentParser(
        description='Parse and deduplicate log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s parse access.log -p '\[.*?\]'
  %(prog)s deduplicate error.log --output results.json
  %(prog)s stats app.log --config ~/.logparse.ini
        '''
    )

    parser.add_argument('--config', type=Path, default=Path.home() / '.logparse.ini',
                        help='Configuration file (default: ~/.logparse.ini)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # parse subcommand
    parse_cmd = subparsers.add_parser('parse', help='Parse log file')
    parse_cmd.add_argument('logfile', type=Path, help='Log file to parse')
    parse_cmd.add_argument('-p', '--pattern', default=None, help='Regex pattern for matching')
    parse_cmd.add_argument('-o', '--output', type=Path, help='Output file')

    # deduplicate subcommand
    dedup_cmd = subparsers.add_parser('deduplicate', help='Find duplicates in log')
    dedup_cmd.add_argument('logfile', type=Path, help='Log file to analyze')
    dedup_cmd.add_argument('--output', type=Path, help='Output file')
    dedup_cmd.add_argument('--format', choices=['json', 'csv'], default='json')

    # stats subcommand
    stats_cmd = subparsers.add_parser('stats', help='Show log statistics')
    stats_cmd.add_argument('logfile', type=Path, help='Log file to analyze')
    stats_cmd.add_argument('--top', type=int, default=10, help='Show top N entries')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)

    if args.verbose:
        print(f'{Colors.BLUE}Config: {config}{Colors.RESET}', file=sys.stderr)

    if args.command == 'parse':
        pattern = args.pattern or config['log_pattern']
        logfile = args.logfile

        if not logfile.exists():
            print(f'{Colors.RED}✗ File not found: {logfile}{Colors.RESET}', file=sys.stderr)
            sys.exit(1)

        entries = parse_logs(logfile, pattern, config['ignore_case'])

        if args.verbose:
            print(f'{Colors.GREEN}Found {len(entries)} entries{Colors.RESET}', file=sys.stderr)

        result = {'entries': entries, 'count': len(entries)}

        if args.output:
            export_json(result, args.output)
        else:
            print(json.dumps(result, indent=2, default=str))

    elif args.command == 'deduplicate':
        logfile = args.logfile

        if not logfile.exists():
            print(f'{Colors.RED}✗ File not found: {logfile}{Colors.RESET}', file=sys.stderr)
            sys.exit(1)

        entries = parse_logs(logfile, config['log_pattern'], config['ignore_case'])
        hash_map = deduplicate_entries(entries, config['hash_algorithm'])

        result = {
            'total_entries': len(entries),
            'unique_entries': len(hash_map),
            'duplicates': {h: len(v) for h, v in hash_map.items() if len(v) > 1}
        }

        if args.output:
            if args.format == 'json':
                export_json(result, args.output)
            else:
                export_csv(result, args.output)
        else:
            print(json.dumps(result, indent=2))

    elif args.command == 'stats':
        logfile = args.logfile

        if not logfile.exists():
            print(f'{Colors.RED}✗ File not found: {logfile}{Colors.RESET}', file=sys.stderr)
            sys.exit(1)

        entries = parse_logs(logfile, config['log_pattern'], config['ignore_case'])
        hash_map = deduplicate_entries(entries, config['hash_algorithm'])

        top_dupes = sorted(hash_map.items(), key=lambda x: len(x[1]), reverse=True)[:args.top]

        print(f'{Colors.BLUE}{"="*60}{Colors.RESET}')
        print(f'{Colors.BLUE}Log Statistics: {logfile}{Colors.RESET}')
        print(f'{Colors.BLUE}{"="*60}{Colors.RESET}')
        print(f'Total entries: {len(entries)}')
        print(f'Unique entries: {len(hash_map)}')
        print(f'Duplicate rate: {100 * (1 - len(hash_map) / max(len(entries), 1)):.1f}%')
        print()
        print(f'{Colors.YELLOW}Top {args.top} most duplicated entries:{Colors.RESET}')

        for i, (h, group) in enumerate(top_dupes, 1):
            sample = group[0]['content'][:50]
            print(f'  {i}. {sample}... ({len(group)} occurrences)')


if __name__ == '__main__':
    main()

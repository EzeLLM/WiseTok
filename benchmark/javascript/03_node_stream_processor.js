const fs = require('fs');
const zlib = require('zlib');
const { Transform, pipeline } = require('stream');
const path = require('path');

class LogProcessor extends Transform {
	constructor(options = {}) {
		super({ ...options, objectMode: true });
		this.lineBuffer = '';
		this.processedCount = 0;
	}

	_transform(chunk, encoding, callback) {
		const data = chunk.toString();
		const lines = (this.lineBuffer + data).split('\n');

		this.lineBuffer = lines.pop();

		for (const line of lines) {
			if (!line.trim()) continue;

			try {
				const parsed = this.parseLine(line);
				this.processedCount++;
				this.push(parsed);
			} catch (err) {
				console.error(`Parse error: ${err.message}`);
			}
		}

		callback();
	}

	_flush(callback) {
		if (this.lineBuffer.trim()) {
			try {
				const parsed = this.parseLine(this.lineBuffer);
				this.push(parsed);
			} catch (err) {
				console.error(`Final line parse error: ${err.message}`);
			}
		}
		callback();
	}

	parseLine(line) {
		const match = line.match(/\[(\d{4}-\d{2}-\d{2}T[\d:\.]+Z)\]\s+(\w+)\s+(.+)/);
		if (!match) throw new Error('Invalid format');

		const [, timestamp, level, message] = match;
		return {
			timestamp: new Date(timestamp),
			level,
			message,
			severity: this.getSeverity(level)
		};
	}

	getSeverity(level) {
		const severities = {
			DEBUG: 0,
			INFO: 1,
			WARN: 2,
			ERROR: 3
		};
		return severities[level] || 0;
	}
}

class AggregationTransform extends Transform {
	constructor(options = {}) {
		super({ ...options, objectMode: true });
		this.window = [];
		this.windowSize = options.windowSize || 10;
	}

	_transform(entry, encoding, callback) {
		this.window.push(entry);

		if (this.window.length >= this.windowSize) {
			const aggregate = {
				startTime: this.window[0].timestamp,
				endTime: this.window[this.window.length - 1].timestamp,
				count: this.window.length,
				errorCount: this.window.filter(e => e.level === 'ERROR').length,
				avgSeverity: this.window.reduce((sum, e) => sum + e.severity, 0) / this.window.length
			};
			this.push(aggregate);
			this.window = [];
		}

		callback();
	}

	_flush(callback) {
		if (this.window.length > 0) {
			const aggregate = {
				startTime: this.window[0].timestamp,
				endTime: this.window[this.window.length - 1].timestamp,
				count: this.window.length,
				errorCount: this.window.filter(e => e.level === 'ERROR').length,
				avgSeverity: this.window.reduce((sum, e) => sum + e.severity, 0) / this.window.length
			};
			this.push(aggregate);
		}
		callback();
	}
}

class JSONWriter extends Transform {
	constructor(options = {}) {
		super({ ...options, objectMode: true });
		this.isFirst = true;
	}

	_transform(obj, encoding, callback) {
		if (!this.isFirst) this.push(',\n');
		this.push(JSON.stringify(obj));
		this.isFirst = false;
		callback();
	}

	_flush(callback) {
		this.push('\n]');
		callback();
	}
}

async function processLogFile(inputPath, outputPath) {
	return new Promise((resolve, reject) => {
		const inputStream = fs.createReadStream(inputPath, { highWaterMark: 64 * 1024 });
		const decompressStream = zlib.createGunzip();
		const logProcessor = new LogProcessor();
		const aggregator = new AggregationTransform({ windowSize: 20 });
		const jsonWriter = new JSONWriter();
		const outputStream = fs.createWriteStream(outputPath);

		let stats = {
			startTime: Date.now(),
			bytesRead: 0,
			entriesProcessed: 0
		};

		inputStream.on('data', (chunk) => {
			stats.bytesRead += chunk.length;
		});

		logProcessor.on('data', () => {
			stats.entriesProcessed++;
		});

		outputStream.on('finish', () => {
			stats.duration = Date.now() - stats.startTime;
			console.log('Processing complete:', stats);
			resolve(stats);
		});

		pipeline(
			inputStream,
			decompressStream,
			logProcessor,
			aggregator,
			jsonWriter,
			outputStream,
			(err) => {
				if (err) {
					console.error('Pipeline error:', err);
					reject(err);
				}
			}
		);
	});
}

async function processMultipleFiles(inputDir, outputDir) {
	try {
		const files = fs.readdirSync(inputDir).filter(f => f.endsWith('.gz'));

		console.log(`Found ${files.length} log files to process`);

		for (const file of files) {
			const inputPath = path.join(inputDir, file);
			const outputPath = path.join(outputDir, file.replace('.gz', '.json'));

			console.log(`Processing: ${file}`);
			await processLogFile(inputPath, outputPath);
		}

		console.log('All files processed');
	} catch (err) {
		console.error('Error:', err);
		process.exit(1);
	}
}

// Usage
if (require.main === module) {
	const inputDir = process.argv[2] || './logs/input';
	const outputDir = process.argv[3] || './logs/output';

	if (!fs.existsSync(outputDir)) {
		fs.mkdirSync(outputDir, { recursive: true });
	}

	processMultipleFiles(inputDir, outputDir);
}

module.exports = { LogProcessor, AggregationTransform, processLogFile };

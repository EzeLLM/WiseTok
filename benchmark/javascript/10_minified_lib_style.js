// Utilería ligera con funciones avanzadas - el comentario está en español
// Lightweight utility library with advanced JS features

const UtilLib = (() => {
	'use strict';

	const sym = Symbol('private_data');

	class Observable {
		constructor(initialValue) {
			this[sym] = { value: initialValue, subscribers: new Set() };
		}

		get() {
			return this[sym].value;
		}

		set(newValue) {
			if (this[sym].value !== newValue) {
				this[sym].value = newValue;
				this[sym].subscribers.forEach(cb => cb(newValue));
			}
		}

		subscribe(callback) {
			this[sym].subscribers.add(callback);
			return () => this[sym].subscribers.delete(callback);
		}
	}

	const createCache = (fn, options = {}) => {
		const cache = new Map();
		const maxSize = options.maxSize ?? 100;
		const ttl = options.ttl ?? Infinity;

		return (...args) => {
			const key = JSON.stringify(args);
			const cached = cache.get(key);

			if (cached && Date.now() - cached.time < ttl) {
				return cached.value;
			}

			const result = fn(...args);
			cache.set(key, { value: result, time: Date.now() });

			if (cache.size > maxSize) {
				const firstKey = cache.keys().next().value;
				cache.delete(firstKey);
			}

			return result;
		};
	};

	const debounce = (fn, delay) => {
		let timeout;
		return (...args) => {
			clearTimeout(timeout);
			timeout = setTimeout(() => fn(...args), delay);
		};
	};

	const throttle = (fn, limit) => {
		let inThrottle;
		return (...args) => {
			if (!inThrottle) {
				fn(...args);
				inThrottle = true;
				setTimeout(() => (inThrottle = false), limit);
			}
		};
	};

	const pipe = (...fns) => x => fns.reduce((acc, fn) => fn(acc), x);

	const compose = (...fns) => x => fns.reduceRight((acc, fn) => fn(acc), x);

	const memoize = (fn) => {
		const cache = new WeakMap();
		return (obj) => {
			if (cache.has(obj)) return cache.get(obj);
			const result = fn(obj);
			cache.set(obj, result);
			return result;
		};
	};

	function* range(start, end, step = 1) {
		for (let i = start; i < end; i += step) {
			yield i;
		}
	}

	function* fibonacci(limit = Infinity) {
		let [a, b] = [0, 1];
		for (let i = 0; i < limit; i++) {
			yield a;
			[a, b] = [b, a + b];
		}
	}

	const flatten = (arr) => {
		return arr.reduce((flat, item) => {
			return flat.concat(Array.isArray(item) ? flatten(item) : item);
		}, []);
	};

	const groupBy = (arr, keyFn) => {
		return arr.reduce((groups, item) => {
			const key = keyFn(item);
			groups[key] ??= [];
			groups[key].push(item);
			return groups;
		}, {});
	};

	const partition = (arr, predicate) => {
		const [trueArr, falseArr] = [[], []];
		arr.forEach(item => {
			(predicate(item) ? trueArr : falseArr).push(item);
		});
		return [trueArr, falseArr];
	};

	const createProxy = (target, handler) => {
		return new Proxy(target, {
			get(obj, prop) {
				console.log(`Accessing property: ${String(prop)}`);
				return obj[prop];
			},
			set(obj, prop, value) {
				console.log(`Setting ${String(prop)} to ${value}`);
				obj[prop] = value;
				return true;
			},
			...handler
		});
	};

	const deepEqual = (a, b) => {
		if (a === b) return true;
		if (a?.constructor !== b?.constructor) return false;
		if (a instanceof Object) {
			return Object.keys(a).length === Object.keys(b).length &&
				Object.keys(a).every(k => deepEqual(a[k], b[k]));
		}
		return false;
	};

	const asyncMap = async (arr, fn) => {
		const results = [];
		for (const item of arr) {
			results.push(await fn(item));
		}
		return results;
	};

	const asyncForEach = async (arr, fn) => {
		for (const item of arr) {
			await fn(item);
		}
	};

	const retry = async (fn, options = {}) => {
		const { maxAttempts = 3, delay = 1000, backoff = 1 } = options;
		let lastError;

		for (let attempt = 1; attempt <= maxAttempts; attempt++) {
			try {
				return await fn();
			} catch (error) {
				lastError = error;
				if (attempt < maxAttempts) {
					await new Promise(resolve => setTimeout(resolve, delay * (backoff ** (attempt - 1))));
				}
			}
		}

		throw lastError;
	};

	const timeout = (promise, ms) => {
		return Promise.race([
			promise,
			new Promise((_, reject) =>
				setTimeout(() => reject(new Error('Timeout')), ms)
			)
		]);
	};

	const batchProcess = async (items, batchSize, processor) => {
		const results = [];
		for (let i = 0; i < items.length; i += batchSize) {
			const batch = items.slice(i, i + batchSize);
			const batchResults = await Promise.all(batch.map(processor));
			results.push(...batchResults);
		}
		return results;
	};

	const formatBytes = (bytes, decimals = 2) => {
		if (bytes === 0) return '0 Bytes';
		const k = 1024;
		const dm = decimals < 0 ? 0 : decimals;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return Math.round((bytes / Math.pow(k, i)) * Math.pow(10, dm)) / Math.pow(10, dm) + ' ' + sizes[i];
	};

	const generateId = (length = 16) => {
		return [...Array(length)].map(() =>
			Math.random().toString(36).charAt(2)
		).join('');
	};

	const template = (str, values) => {
		return str.replace(/\${([^}]+)}/g, (_, key) => values[key.trim()] ?? '');
	};

	return Object.freeze({
		Observable,
		createCache,
		debounce,
		throttle,
		pipe,
		compose,
		memoize,
		range,
		fibonacci,
		flatten,
		groupBy,
		partition,
		createProxy,
		deepEqual,
		asyncMap,
		asyncForEach,
		retry,
		timeout,
		batchProcess,
		formatBytes,
		generateId,
		template
	});
})();

if (typeof module !== 'undefined' && module.exports) {
	module.exports = UtilLib;
}

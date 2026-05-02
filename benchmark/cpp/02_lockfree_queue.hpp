#ifndef LOCKFREE_QUEUE_HPP
#define LOCKFREE_QUEUE_HPP

#include <atomic>
#include <memory>
#include <thread>
#include <cstring>
#include <stdexcept>

template <typename T>
class LockFreeQueue {
	struct Node {
		T value;
		std::atomic<Node*> next{nullptr};
		std::atomic<int> hazard_count{0};
	};

	std::atomic<Node*> head;
	std::atomic<Node*> tail;
	thread_local static Node* thread_hazard_ptr;

public:
	LockFreeQueue() {
		auto sentinel = new Node();
		head.store(sentinel, std::memory_order_release);
		tail.store(sentinel, std::memory_order_release);
	}

	~LockFreeQueue() {
		Node* current = head.load(std::memory_order_acquire);
		while (current != nullptr) {
			Node* next = current->next.load(std::memory_order_acquire);
			delete current;
			current = next;
		}
	}

	void enqueue(const T& value) {
		auto new_node = new Node();
		new_node->value = value;

		Node* old_tail;
		Node* next;

		while (true) {
			old_tail = tail.load(std::memory_order_acquire);
			next = old_tail->next.load(std::memory_order_acquire);

			if (old_tail != tail.load(std::memory_order_acquire)) {
				continue;
			}

			if (next == nullptr) {
				if (old_tail->next.compare_exchange_weak(
					next, new_node,
					std::memory_order_release,
					std::memory_order_acquire)) {
					tail.compare_exchange_strong(
						old_tail, new_node,
						std::memory_order_release,
						std::memory_order_acquire);
					return;
				}
			} else {
				tail.compare_exchange_strong(
					old_tail, next,
					std::memory_order_release,
					std::memory_order_acquire);
			}
		}
	}

	bool dequeue(T& value) {
		Node* old_head;
		Node* next;

		while (true) {
			old_head = head.load(std::memory_order_acquire);
			thread_hazard_ptr = old_head;

			if (old_head != head.load(std::memory_order_acquire)) {
				continue;
			}

			next = old_head->next.load(std::memory_order_acquire);
			Node* current_tail = tail.load(std::memory_order_acquire);

			if (old_head == current_tail) {
				if (next == nullptr) {
					thread_hazard_ptr = nullptr;
					return false;
				}
				tail.compare_exchange_strong(
					current_tail, next,
					std::memory_order_release,
					std::memory_order_acquire);
			} else {
				if (next == nullptr) {
					continue;
				}

				value = next->value;

				if (head.compare_exchange_weak(
					old_head, next,
					std::memory_order_release,
					std::memory_order_acquire)) {
					thread_hazard_ptr = nullptr;
					retire_node(old_head);
					return true;
				}
			}
		}
	}

	bool try_dequeue(T& value) {
		if (!dequeue(value)) {
			return false;
		}
		return true;
	}

	bool empty() const {
		Node* h = head.load(std::memory_order_acquire);
		Node* t = tail.load(std::memory_order_acquire);
		Node* next = h->next.load(std::memory_order_acquire);
		return h == t && next == nullptr;
	}

	void retire_node(Node* node) {
		node->hazard_count.fetch_sub(1, std::memory_order_release);
		if (node->hazard_count.load(std::memory_order_acquire) == 0) {
			delete node;
		}
	}

	class Iterator {
		Node* current;

	public:
		Iterator(Node* node) : current(node) {}

		Iterator& operator++() {
			if (current) {
				current = current->next.load(std::memory_order_acquire);
			}
			return *this;
		}

		T& operator*() const {
			return current->value;
		}

		bool operator!=(const Iterator& other) const {
			return current != other.current;
		}
	};

	Iterator begin() {
		return Iterator(head.load(std::memory_order_acquire));
	}

	Iterator end() {
		return Iterator(nullptr);
	}
};

template <typename T>
thread_local typename LockFreeQueue<T>::Node*
	LockFreeQueue<T>::thread_hazard_ptr = nullptr;

template <typename Key, typename Value>
class LockFreeHashMap {
	static constexpr std::size_t CAPACITY = 65536;

	struct Entry {
		std::atomic<Key> key{Key()};
		std::atomic<Value> value{Value()};
		std::atomic<bool> occupied{false};
	};

	std::unique_ptr<Entry[]> table;

public:
	LockFreeHashMap() : table(std::make_unique<Entry[]>(CAPACITY)) {}

	void insert(const Key& key, const Value& value) {
		std::size_t hash = std::hash<Key>()(key);
		std::size_t idx = hash % CAPACITY;

		while (true) {
			Entry& entry = table[idx];
			bool expected_occupied = false;

			if (entry.occupied.compare_exchange_weak(
				expected_occupied, true,
				std::memory_order_release,
				std::memory_order_acquire)) {
				entry.key.store(key, std::memory_order_release);
				entry.value.store(value, std::memory_order_release);
				return;
			}

			if (entry.key.load(std::memory_order_acquire) == key) {
				entry.value.store(value, std::memory_order_release);
				return;
			}

			idx = (idx + 1) % CAPACITY;
		}
	}

	bool lookup(const Key& key, Value& result) {
		std::size_t hash = std::hash<Key>()(key);
		std::size_t idx = hash % CAPACITY;

		for (std::size_t i = 0; i < CAPACITY; ++i) {
			Entry& entry = table[idx];
			if (entry.occupied.load(std::memory_order_acquire)) {
				if (entry.key.load(std::memory_order_acquire) == key) {
					result = entry.value.load(std::memory_order_acquire);
					return true;
				}
			}
			idx = (idx + 1) % CAPACITY;
		}
		return false;
	}
};

#endif

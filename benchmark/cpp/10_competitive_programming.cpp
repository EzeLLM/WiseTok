#include<bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<ll, ll> pll;
typedef vector<ll> vll;

#define fast ios_base::sync_with_stdio(0); cin.tie(0);
#define pb push_back
#define all(x) x.begin(),x.end()
#define sz(x) (ll)x.size()
#define MOD 1e9+7
#define INF 1e18

class SegmentTree {
private:
	vll tree, arr;
	ll n;

	void build(ll node, ll start, ll end) {
		if (start == end) {
			tree[node] = arr[start];
		} else {
			ll mid = (start + end) / 2;
			build(2 * node, start, mid);
			build(2 * node + 1, mid + 1, end);
			tree[node] = min(tree[2 * node], tree[2 * node + 1]);
		}
	}

	void update(ll node, ll start, ll end, ll idx, ll val) {
		if (start == end) {
			tree[node] = val;
		} else {
			ll mid = (start + end) / 2;
			if (idx <= mid) {
				update(2 * node, start, mid, idx, val);
			} else {
				update(2 * node + 1, mid + 1, end, idx, val);
			}
			tree[node] = min(tree[2 * node], tree[2 * node + 1]);
		}
	}

	ll query(ll node, ll start, ll end, ll l, ll r) {
		if (r < start || l > end) return INF;
		if (l <= start && end <= r) return tree[node];
		ll mid = (start + end) / 2;
		return min(query(2 * node, start, mid, l, r),
		           query(2 * node + 1, mid + 1, end, l, r));
	}

public:
	SegmentTree(vll& a) : arr(a) {
		n = sz(a);
		tree.resize(4 * n);
		build(1, 0, n - 1);
	}

	void update(ll idx, ll val) {
		update(1, 0, n - 1, idx, val);
	}

	ll query(ll l, ll r) {
		return query(1, 0, n - 1, l, r);
	}
};

class DSU {
private:
	vll parent, rnk;

public:
	DSU(ll n) : parent(n), rnk(n, 0) {
		iota(all(parent), 0);
	}

	ll find(ll x) {
		if (parent[x] != x) {
			parent[x] = find(parent[x]);
		}
		return parent[x];
	}

	bool unite(ll x, ll y) {
		x = find(x);
		y = find(y);
		if (x == y) return false;
		if (rnk[x] < rnk[y]) swap(x, y);
		parent[y] = x;
		if (rnk[x] == rnk[y]) rnk[x]++;
		return true;
	}

	bool connected(ll x, ll y) {
		return find(x) == find(y);
	}
};

int main() {
	fast

	ll t;
	cin >> t;

	while (t--) {
		ll n, q;
		cin >> n >> q;

		vll a(n);
		for (auto& x : a) cin >> x;

		SegmentTree st(a);

		while (q--) {
			ll op;
			cin >> op;

			if (op == 1) {
				ll idx, val;
				cin >> idx >> val;
				st.update(idx - 1, val);
			} else {
				ll l, r;
				cin >> l >> r;
				cout << st.query(l - 1, r - 1) << "\n";
			}
		}
	}

	return 0;
}

ll gcd(ll a, ll b) {
	return b ? gcd(b, a % b) : a;
}

ll power(ll a, ll b, ll mod) {
	ll res = 1;
	a %= mod;
	while (b > 0) {
		if (b & 1) res = (res * a) % mod;
		a = (a * a) % mod;
		b >>= 1;
	}
	return res;
}

ll modInv(ll a, ll mod) {
	return power(a, mod - 2, mod);
}

template<typename T>
void read(T& x) {
	cin >> x;
}

template<typename T, typename... Args>
void read(T& x, Args&... args) {
	cin >> x;
	read(args...);
}

template<typename T>
void readVec(vector<T>& v, ll n) {
	v.resize(n);
	for (auto& x : v) cin >> x;
}

ll countBits(ll x) {
	return __builtin_popcountll(x);
}

ll ctz(ll x) {
	return __builtin_ctzll(x);
}

template<typename Func>
void forRange(ll l, ll r, Func f) {
	for (ll i = l; i <= r; ++i) {
		f(i);
	}
}

auto kmp(string s) {
	ll n = sz(s);
	vll pi(n);
	for (ll i = 1; i < n; ++i) {
		ll j = pi[i - 1];
		while (j > 0 && s[i] != s[j]) j = pi[j - 1];
		if (s[i] == s[j]) j++;
		pi[i] = j;
	}
	return pi;
}

struct Edge {
	ll u, v, w;
	bool operator<(const Edge& e) const {
		return w < e.w;
	}
};

ll kruskal(vll& edges, ll n) {
	DSU dsu(n);
	ll mst_weight = 0;
	for (auto& e : edges) {
		if (dsu.unite(e.u, e.v)) {
			mst_weight += e.w;
		}
	}
	return mst_weight;
}

struct Node {
	ll val, prio;
	Node *l, *r;
	Node(ll v) : val(v), prio(rand()), l(nullptr), r(nullptr) {}
};

ll merge(Node*& t, Node* l, Node* r) {
	if (!l) { t = r; return 0; }
	if (!r) { t = l; return 0; }
	if (l->prio > r->prio) {
		return merge(l->r, l->r, r);
	} else {
		return merge(r->l, l, r->l);
	}
}

void split(Node* t, Node*& l, Node*& r, ll k) {
	if (!t) { l = r = nullptr; return; }
	if (t->val <= k) {
		l = t;
		split(t->r, t->r, r, k);
	} else {
		r = t;
		split(t->l, l, t->l, k);
	}
}

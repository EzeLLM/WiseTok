#include <iostream>
#include <type_traits>
#include <tuple>
#include <memory>
#include <functional>

template <typename T, typename Enable = void>
struct IsContainer : std::false_type {};

template <typename T>
struct IsContainer<T, std::void_t<
    typename T::value_type,
    typename T::iterator,
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : std::true_type {};

template <typename T>
constexpr bool is_container_v = IsContainer<T>::value;

template <int N>
struct Factorial {
    static constexpr int value = N * Factorial<N - 1>::value;
};

template <>
struct Factorial<0> {
    static constexpr int value = 1;
};

template <typename... Args>
struct TypeCount {
    static constexpr std::size_t value = sizeof...(Args);
};

template <typename First, typename... Rest>
struct FirstType {
    using type = First;
};

template <std::size_t I, typename T, typename... Args>
struct TypeAt {
    using type = typename TypeAt<I - 1, Args...>::type;
};

template <typename T, typename... Args>
struct TypeAt<0, T, Args...> {
    using type = T;
};

template <typename T, typename U>
struct IsSame : std::false_type {};

template <typename T>
struct IsSame<T, T> : std::true_type {};

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
T add(T a, T b) { return a + b; }

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
T multiply(T a, T b) { return a * b; }

template <typename T>
class SmartContainer {
    std::unique_ptr<T[]> data;
    std::size_t sz;

public:
    SmartContainer(std::size_t n) : data(std::make_unique<T[]>(n)), sz(n) {}

    template <typename U = T, std::enable_if_t<std::is_default_constructible_v<U>, int> = 0>
    void reset() {
        for (std::size_t i = 0; i < sz; ++i)
            data[i] = U();
    }

    T& at(std::size_t idx) { return data[idx]; }
    std::size_t size() const { return sz; }
};

template <typename Func, typename... Args>
auto invoke_if_callable(Func&& f, Args&&... args)
    -> std::enable_if_t<std::is_invocable_v<Func, Args...>,
                       std::invoke_result_t<Func, Args...>> {
    return std::invoke(f, std::forward<Args>(args)...);
}

namespace detail {
    template <typename Tuple, typename Func, std::size_t... I>
    void apply_impl(Tuple& t, Func&& f, std::index_sequence<I...>) {
        ((void)f(std::get<I>(t)), ...);
    }
}

template <typename Tuple, typename Func>
void apply_all(Tuple& t, Func&& f) {
    detail::apply_impl(t, std::forward<Func>(f),
        std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <typename... Ts>
class VariantLike {
    std::tuple<Ts...> data;

public:
    template <typename F>
    void for_each(F&& func) {
        apply_all(data, std::forward<F>(func));
    }
};

constexpr bool is_power_of_two(unsigned long long x) {
    return x > 0 && (x & (x - 1)) == 0;
}

constexpr int fib(int n) {
    return n <= 1 ? n : fib(n - 1) + fib(n - 2);
}

template <typename T, std::enable_if_t<std::is_array_v<T>, int> = 0>
std::size_t array_size(const T&) {
    return std::extent_v<T>;
}

int main() {
    static_assert(Factorial<5>::value == 120);
    static_assert(is_power_of_two(16));
    static_assert(fib(10) == 55);
    static_assert(TypeCount<int, double, char>::value == 3);
    static_assert(std::is_same_v<typename FirstType<int, double>::type, int>);
    static_assert(std::is_same_v<typename TypeAt<1, int, double, char>::type, double>);

    SmartContainer<int> container(100);
    container.reset<int>();

    std::cout << "Template metaprogramming checks passed\n";
    return 0;
}

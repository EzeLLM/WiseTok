/*
 * common.h - Common definitions and utilities header
 *
 * Provides type definitions, macros, and function prototypes used
 * throughout the application. Uses X-macro pattern for boilerplate
 * generation and careful include guards.
 */

#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version information */
#define APP_VERSION "1.0.0"
#define APP_BUILD_DATE __DATE__

/* Platform detection macros */
#ifdef __linux__
  #define PLATFORM_LINUX 1
  #define PLATFORM_NAME "Linux"
#elif __APPLE__
  #define PLATFORM_MACOS 1
  #define PLATFORM_NAME "macOS"
#elif _WIN32
  #define PLATFORM_WINDOWS 1
  #define PLATFORM_NAME "Windows"
#else
  #define PLATFORM_UNKNOWN 1
  #define PLATFORM_NAME "Unknown"
#endif

/* Compiler attributes and hints */
#if defined(__GNUC__) || defined(__clang__)
  #define LIKELY(x) __builtin_expect(!!(x), 1)
  #define UNLIKELY(x) __builtin_expect(!!(x), 0)
  #define NORETURN __attribute__((noreturn))
  #define PURE __attribute__((pure))
  #define CONST __attribute__((const))
  #define UNUSED __attribute__((unused))
  #define ALIGNED(n) __attribute__((aligned(n)))
  #define PACKED __attribute__((packed))
#else
  #define LIKELY(x) (x)
  #define UNLIKELY(x) (x)
  #define NORETURN
  #define PURE
  #define CONST
  #define UNUSED
  #define ALIGNED(n)
  #define PACKED
#endif

/* Utility macros */
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, lo, hi) MIN(MAX(x, lo), hi)
#define SWAP(a, b) do { typeof(a) tmp = (a); (a) = (b); (b) = tmp; } while(0)
#define CONTAINER_OF(ptr, type, member) ((type *)((char *)(ptr) - offsetof(type, member)))

/* Error handling macros */
#define CHECK(cond) assert(cond)
#define CHECKF(cond, fmt, ...) \
  if (UNLIKELY(!(cond))) { \
    fprintf(stderr, "CHECK FAILED: " fmt "\n", ##__VA_ARGS__); \
    abort(); \
  }

#define GOTO_IF_ERR(cond, label) \
  do { \
    if (UNLIKELY(cond)) { \
      goto label; \
    } \
  } while (0)

#define RETURN_IF_ERR(cond, val) \
  do { \
    if (UNLIKELY(cond)) { \
      return (val); \
    } \
  } while (0)

/* Type definitions */
typedef int8_t s8;
typedef uint8_t u8;
typedef int16_t s16;
typedef uint16_t u16;
typedef int32_t s32;
typedef uint32_t u32;
typedef int64_t s64;
typedef uint64_t u64;
typedef float f32;
typedef double f64;

typedef struct {
  void *ptr;
  size_t len;
  size_t capacity;
} buffer_t;

typedef struct {
  const char *str;
  size_t len;
} string_view_t;

typedef struct {
  s32 x;
  s32 y;
} point_t;

typedef struct {
  s32 x;
  s32 y;
  s32 w;
  s32 h;
} rect_t;

typedef struct {
  f32 r;
  f32 g;
  f32 b;
  f32 a;
} color_t;

/* X-Macro: Define common error codes */
#define ENUM_ERRORS(X) \
  X(ERR_OK, "Success") \
  X(ERR_NOMEM, "Out of memory") \
  X(ERR_INVAL, "Invalid argument") \
  X(ERR_IO, "I/O error") \
  X(ERR_TIMEOUT, "Operation timeout") \
  X(ERR_NOTFOUND, "Resource not found") \
  X(ERR_PERM, "Permission denied")

#define DEFINE_ENUM(name, desc) name,
typedef enum {
  ENUM_ERRORS(DEFINE_ENUM)
  ERR_COUNT
} error_t;

#define DEFINE_STR(name, desc) desc,
static const char *error_strings[] = {
  ENUM_ERRORS(DEFINE_STR)
};

#define DEFINE_NAME(name, desc) #name,
static const char *error_names[] = {
  ENUM_ERRORS(DEFINE_NAME)
};

/* Parameterized macros for common patterns */
#define DEFINE_STACK(name, type) \
  typedef struct { \
    type *data; \
    size_t size; \
    size_t capacity; \
  } name##_t; \
  \
  static inline name##_t *name##_create(size_t cap) { \
    name##_t *s = malloc(sizeof(name##_t)); \
    if (!s) return NULL; \
    s->data = malloc(cap * sizeof(type)); \
    if (!s->data) { free(s); return NULL; } \
    s->size = 0; \
    s->capacity = cap; \
    return s; \
  } \
  \
  static inline void name##_push(name##_t *s, type val) { \
    if (UNLIKELY(s->size >= s->capacity)) { \
      s->capacity *= 2; \
      s->data = realloc(s->data, s->capacity * sizeof(type)); \
    } \
    s->data[s->size++] = val; \
  } \
  \
  static inline type name##_pop(name##_t *s) { \
    return s->data[--s->size]; \
  } \
  \
  static inline type name##_peek(name##_t *s) { \
    return s->data[s->size - 1]; \
  } \
  \
  static inline void name##_destroy(name##_t *s) { \
    free(s->data); free(s); \
  }

/* Function prototypes */

/**
 * error_str - Get human-readable error message
 * @e: Error code
 *
 * Returns a string describing the error code.
 */
const char *error_str(error_t e);

/**
 * buffer_alloc - Allocate a new buffer
 * @cap: Initial capacity in bytes
 *
 * Returns a newly allocated buffer, or NULL on error.
 */
buffer_t *buffer_alloc(size_t cap);

/**
 * buffer_append - Append data to buffer
 * @buf: Buffer to append to
 * @data: Data to append
 * @len: Length of data
 *
 * Appends len bytes of data to the buffer, growing capacity as needed.
 */
error_t buffer_append(buffer_t *buf, const void *data, size_t len);

/**
 * buffer_free - Deallocate buffer
 * @buf: Buffer to free
 */
void buffer_free(buffer_t *buf);

/**
 * string_view_eq - Compare two string views
 * @a: First string view
 * @b: Second string view
 *
 * Returns true if the strings are equal.
 */
bool string_view_eq(string_view_t a, string_view_t b) PURE;

/**
 * rect_contains - Check if point is within rectangle
 * @r: Rectangle
 * @p: Point
 *
 * Returns true if the point is inside or on the rectangle boundary.
 */
bool rect_contains(rect_t r, point_t p) PURE;

/**
 * color_blend - Blend two colors
 * @c1: First color
 * @c2: Second color
 * @alpha: Blend factor (0.0 to 1.0)
 *
 * Returns a blended color between c1 and c2.
 */
color_t color_blend(color_t c1, color_t c2, f32 alpha) CONST;

#ifdef __cplusplus
}
#endif

#endif /* COMMON_H */

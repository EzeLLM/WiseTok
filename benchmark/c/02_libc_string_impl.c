/*
 * string_impl.c - High-performance string library implementations
 *
 * Custom implementations of standard C string functions optimized for
 * performance using word-at-a-time algorithms and register hints.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define WORD_SIZE (sizeof(unsigned long))
#define ALIGNED(p) (((uintptr_t)(p) & (WORD_SIZE - 1)) == 0)
#define REPEAT_BYTE(x) ((x UL) / 0xFF)
#define ONEBYTE_MASK ((unsigned long) -1 / 0xFF)

/* Portable zero-byte detection using word operations */
static inline int has_zero_byte(unsigned long word)
{
  return ((word - ONEBYTE_MASK) & ~word & (ONEBYTE_MASK << 7));
}

/**
 * string_memcpy - Copy memory regions, optimized for speed
 * @restrict dest: Destination buffer (non-overlapping with src)
 * @restrict src: Source buffer
 * @n: Number of bytes to copy
 *
 * Copies memory using word-at-a-time reads for aligned buffers,
 * falling back to byte-by-byte for unaligned regions.
 */
void *string_memcpy(void * restrict dest,
                    const void * restrict src,
                    size_t n)
{
  unsigned char *d = (unsigned char *)dest;
  const unsigned char *s = (const unsigned char *)src;
  unsigned long *ld;
  const unsigned long *ls;
  size_t i;

  /* Handle leading bytes until destination is aligned */
  if (!ALIGNED(d) || !ALIGNED(s)) {
    i = 0;
    while (i < n && !ALIGNED(d)) {
      d[i] = s[i];
      i++;
    }
    if (i < n) {
      return string_memcpy(&d[i], &s[i], n - i);
    }
    return dest;
  }

  /* Copy word-aligned blocks */
  ld = (unsigned long *)d;
  ls = (const unsigned long *)s;
  for (i = 0; i + WORD_SIZE <= n; i += WORD_SIZE) {
    *ld++ = *ls++;
  }

  /* Copy remaining bytes */
  d = (unsigned char *)ld;
  s = (const unsigned char *)ls;
  for (; i < n; i++) {
    d[i - (i & ~(WORD_SIZE - 1))] = s[i - (i & ~(WORD_SIZE - 1))];
  }

  return dest;
}

/**
 * string_strlen - Compute string length
 * @s: Null-terminated string
 *
 * Returns the length of the string, excluding the null terminator.
 * Optimized using word-at-a-time zero detection.
 */
size_t string_strlen(const char *s)
{
  const unsigned char *p = (const unsigned char *)s;
  const unsigned long *lp;
  unsigned long word;

  /* Align pointer to word boundary */
  while ((uintptr_t)p & (WORD_SIZE - 1)) {
    if (*p == '\0') {
      return (const char *)p - s;
    }
    p++;
  }

  /* Search word-aligned blocks */
  lp = (const unsigned long *)p;
  while (1) {
    word = *lp++;
    if (has_zero_byte(word)) {
      break;
    }
  }

  /* Back up and find the exact null byte */
  p = (const unsigned char *)(lp - 1);
  while (*p != '\0') {
    p++;
  }

  return (const char *)p - s;
}

/**
 * string_strcmp - Compare two strings
 * @s1: First string
 * @s2: Second string
 *
 * Compares strings lexicographically.
 * Returns 0 if equal, negative if s1 < s2, positive if s1 > s2.
 */
int string_strcmp(const char * restrict s1, const char * restrict s2)
{
  register const unsigned char *p1 = (const unsigned char *)s1;
  register const unsigned char *p2 = (const unsigned char *)s2;

  while (*p1 && *p1 == *p2) {
    p1++;
    p2++;
  }

  return (int)(*p1) - (int)(*p2);
}

/**
 * string_strncpy - Copy n bytes from source to destination
 * @restrict dest: Destination buffer
 * @restrict src: Source string
 * @n: Maximum number of bytes to copy
 *
 * Copies up to n bytes from src to dest. If src is shorter than n,
 * the destination is padded with null bytes.
 */
char *string_strncpy(char * restrict dest,
                     const char * restrict src,
                     size_t n)
{
  register char *d = dest;
  register const char *s = src;
  size_t i;

  for (i = 0; i < n && s[i] != '\0'; i++) {
    d[i] = s[i];
  }

  while (i < n) {
    d[i++] = '\0';
  }

  return dest;
}

/**
 * string_memcmp - Compare memory regions
 * @s1: First buffer
 * @s2: Second buffer
 * @n: Number of bytes to compare
 *
 * Compares two memory regions byte by byte.
 * Returns 0 if equal, negative if s1 < s2, positive if s1 > s2.
 */
int string_memcmp(const void *s1, const void *s2, size_t n)
{
  register const unsigned char *p1 = (const unsigned char *)s1;
  register const unsigned char *p2 = (const unsigned char *)s2;
  size_t i;

  for (i = 0; i < n; i++) {
    if (p1[i] != p2[i]) {
      return (int)(p1[i]) - (int)(p2[i]);
    }
  }

  return 0;
}

/**
 * string_memmove - Move memory regions (handles overlap)
 * @dest: Destination buffer
 * @src: Source buffer
 * @n: Number of bytes to move
 *
 * Like memcpy but handles overlapping source and destination.
 */
void *string_memmove(void *dest, const void *src, size_t n)
{
  unsigned char *d = (unsigned char *)dest;
  const unsigned char *s = (const unsigned char *)src;

  /* If source is before destination, copy backwards */
  if (s < d && s + n > d) {
    d += n;
    s += n;
    while (n-- > 0) {
      *--d = *--s;
    }
    return dest;
  }

  /* Otherwise forward copy is safe */
  return string_memcpy(dest, src, n);
}

/**
 * string_memset - Fill memory with a byte value
 * @s: Buffer to fill
 * @c: Byte value to replicate
 * @n: Number of bytes to fill
 *
 * Sets n bytes of memory to the value c.
 */
void *string_memset(void *s, int c, size_t n)
{
  register unsigned char *p = (unsigned char *)s;
  register unsigned char byte = (unsigned char)c;
  register unsigned long *lp;
  unsigned long pattern;
  size_t i;

  /* Fill unaligned leading bytes */
  while (n > 0 && !ALIGNED(p)) {
    *p++ = byte;
    n--;
  }

  if (n == 0) {
    return s;
  }

  /* Fill word-aligned blocks */
  pattern = REPEAT_BYTE(byte);
  lp = (unsigned long *)p;
  for (i = 0; i + WORD_SIZE <= n; i += WORD_SIZE) {
    *lp++ = pattern;
  }

  /* Fill remaining bytes */
  p = (unsigned char *)lp;
  n -= i;
  while (n-- > 0) {
    *p++ = byte;
  }

  return s;
}

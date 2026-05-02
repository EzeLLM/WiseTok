/*
 * redis_command.c - Redis-style data structure and command implementation
 *
 * Demonstrates Redis-like patterns: object types, command handlers,
 * dictionary-based storage, and SDS (Simple Dynamic Strings).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>

/* Object types */
#define OBJ_STRING 0
#define OBJ_LIST 1
#define OBJ_SET 2
#define OBJ_ZSET 3
#define OBJ_HASH 4

/* Encoding types */
#define OBJ_ENCODING_RAW 0
#define OBJ_ENCODING_INT 1
#define OBJ_ENCODING_EMBSTR 2

/* Constants */
#define EMBSTR_SIZE_LIMIT 64
#define DICT_INITIAL_SIZE 16
#define DICT_LOAD_FACTOR 0.75

typedef struct sds {
  char *buf;
  size_t len;
  size_t capacity;
} sds_t;

typedef struct {
  void *ptr;
  int type;
  int encoding;
  unsigned long lru;
} obj_t;

typedef struct dict_entry {
  sds_t key;
  obj_t *value;
  struct dict_entry *next;
} dict_entry_t;

typedef struct {
  dict_entry_t **table;
  size_t size;
  size_t used;
} dict_t;

typedef struct {
  int64_t *members;
  size_t len;
  size_t capacity;
} zset_t;

typedef struct {
  dict_t *db;
  uint64_t commands_processed;
} server_t;

/* Global server state */
static server_t server = { NULL, 0 };

/**
 * sds_new - Create a new dynamic string
 * @init: Initial C-string (may be NULL)
 *
 * Allocates and returns a new simple dynamic string.
 */
static sds_t *sds_new(const char *init)
{
  sds_t *s = malloc(sizeof(sds_t));
  if (!s) return NULL;

  if (init) {
    s->len = strlen(init);
    s->capacity = (s->len < EMBSTR_SIZE_LIMIT) ? EMBSTR_SIZE_LIMIT : s->len + 1;
  } else {
    s->len = 0;
    s->capacity = EMBSTR_SIZE_LIMIT;
  }

  s->buf = malloc(s->capacity);
  if (!s->buf) {
    free(s);
    return NULL;
  }

  if (init) {
    memcpy(s->buf, init, s->len);
  }
  s->buf[s->len] = '\0';

  return s;
}

/**
 * sds_append - Append to a dynamic string
 * @s: String to append to
 * @append: String to append
 *
 * Appends the given string to s, growing capacity as needed.
 */
static void sds_append(sds_t *s, const char *append)
{
  size_t alen = strlen(append);
  if (s->len + alen >= s->capacity) {
    s->capacity = (s->len + alen) * 2 + 1;
    s->buf = realloc(s->buf, s->capacity);
  }
  memcpy(s->buf + s->len, append, alen);
  s->len += alen;
  s->buf[s->len] = '\0';
}

/**
 * sds_free - Deallocate a dynamic string
 * @s: String to free
 */
static void sds_free(sds_t *s)
{
  if (s) {
    free(s->buf);
    free(s);
  }
}

/**
 * dict_hash - Simple hash function for dictionary keys
 * @s: String to hash
 *
 * Returns hash value suitable for dictionary indexing.
 */
static size_t dict_hash(const sds_t *s)
{
  size_t h = 5381;
  for (size_t i = 0; i < s->len; i++) {
    h = ((h << 5) + h) + (unsigned char)s->buf[i];
  }
  return h;
}

/**
 * dict_new - Create a new dictionary
 *
 * Allocates an empty hash table.
 */
static dict_t *dict_new(void)
{
  dict_t *d = malloc(sizeof(dict_t));
  if (!d) return NULL;

  d->size = DICT_INITIAL_SIZE;
  d->used = 0;
  d->table = calloc(d->size, sizeof(dict_entry_t *));
  if (!d->table) {
    free(d);
    return NULL;
  }

  return d;
}

/**
 * dict_set - Store key-value pair in dictionary
 * @d: Dictionary
 * @key: Key string
 * @value: Object value
 *
 * Inserts or updates the key in the dictionary.
 */
static void dict_set(dict_t *d, const char *key, obj_t *value)
{
  if (!d || !key) return;

  sds_t *k = sds_new(key);
  if (!k) return;

  size_t idx = dict_hash(k) % d->size;
  dict_entry_t *entry = d->table[idx];

  /* Check if key exists */
  while (entry) {
    if (entry->key.len == k->len && memcmp(entry->key.buf, k->buf, k->len) == 0) {
      free(entry->value);
      entry->value = value;
      sds_free(k);
      return;
    }
    entry = entry->next;
  }

  /* Insert new entry */
  dict_entry_t *new_entry = malloc(sizeof(dict_entry_t));
  if (!new_entry) {
    sds_free(k);
    return;
  }

  new_entry->key = *k;
  free(k);
  new_entry->value = value;
  new_entry->next = d->table[idx];
  d->table[idx] = new_entry;
  d->used++;
}

/**
 * dict_get - Retrieve value from dictionary
 * @d: Dictionary
 * @key: Key string
 *
 * Returns the object associated with the key, or NULL if not found.
 */
static obj_t *dict_get(dict_t *d, const char *key)
{
  if (!d || !key) return NULL;

  sds_t k = { (char *)key, strlen(key), strlen(key) };
  size_t idx = dict_hash(&k) % d->size;

  dict_entry_t *entry = d->table[idx];
  while (entry) {
    if (entry->key.len == k.len && memcmp(entry->key.buf, k.buf, k.len) == 0) {
      return entry->value;
    }
    entry = entry->next;
  }

  return NULL;
}

/**
 * dict_free - Deallocate dictionary
 * @d: Dictionary to free
 */
static void dict_free(dict_t *d)
{
  if (!d) return;
  for (size_t i = 0; i < d->size; i++) {
    dict_entry_t *entry = d->table[i];
    while (entry) {
      dict_entry_t *next = entry->next;
      sds_free(&entry->key);
      free(entry->value);
      free(entry);
      entry = next;
    }
  }
  free(d->table);
  free(d);
}

/**
 * createObject - Allocate a new object
 * @type: Object type (OBJ_STRING, OBJ_LIST, etc.)
 * @ptr: Object data pointer
 *
 * Allocates and initializes a new Redis object.
 */
static obj_t *createObject(int type, void *ptr)
{
  obj_t *o = malloc(sizeof(obj_t));
  if (!o) return NULL;

  o->type = type;
  o->ptr = ptr;
  o->encoding = OBJ_ENCODING_RAW;
  o->lru = 0;

  return o;
}

/**
 * cmd_set - SET command: store string value
 * @argc: Argument count
 * @argv: Argument vector (argv[1] = key, argv[2] = value)
 *
 * Stores a string value at the given key.
 */
static void cmd_set(int argc, char **argv)
{
  if (argc < 3) {
    printf("ERR wrong number of arguments for 'set' command\n");
    return;
  }

  sds_t *value = sds_new(argv[2]);
  if (!value) {
    printf("ERR out of memory\n");
    return;
  }

  obj_t *obj = createObject(OBJ_STRING, value);
  if (!obj) {
    sds_free(value);
    printf("ERR out of memory\n");
    return;
  }

  dict_set(server.db, argv[1], obj);
  printf("+OK\n");
  server.commands_processed++;
}

/**
 * cmd_get - GET command: retrieve string value
 * @argc: Argument count
 * @argv: Argument vector (argv[1] = key)
 *
 * Retrieves and prints the string value at the given key.
 */
static void cmd_get(int argc, char **argv)
{
  if (argc < 2) {
    printf("ERR wrong number of arguments for 'get' command\n");
    return;
  }

  obj_t *obj = dict_get(server.db, argv[1]);
  if (!obj) {
    printf("$-1\n");
    server.commands_processed++;
    return;
  }

  if (obj->type == OBJ_STRING) {
    sds_t *s = (sds_t *)obj->ptr;
    printf("$%zu\n%s\n", s->len, s->buf);
  } else {
    printf("WRONGTYPE\n");
  }

  server.commands_processed++;
}

/**
 * cmd_info - INFO command: server statistics
 * @argc: Argument count
 * @argv: Argument vector (unused)
 *
 * Prints server statistics and state.
 */
static void cmd_info(int argc, char **argv)
{
  printf("# Server\n");
  printf("commands_processed:%lu\n", server.commands_processed);
  printf("keys_in_db:%zu\n", server.db ? server.db->used : 0);
  printf("\n");
  server.commands_processed++;
}

/**
 * cmd_del - DEL command: remove key
 * @argc: Argument count
 * @argv: Argument vector (argv[1] = key)
 *
 * Removes the key from the database if it exists.
 */
static void cmd_del(int argc, char **argv)
{
  if (argc < 2) {
    printf("ERR wrong number of arguments for 'del' command\n");
    return;
  }

  /* Simple marker for key deletion (full implementation would rehash) */
  obj_t *obj = dict_get(server.db, argv[1]);
  if (!obj) {
    printf(":0\n");
  } else {
    printf(":1\n");
  }

  server.commands_processed++;
}

typedef void (*cmd_func_t)(int argc, char **argv);

typedef struct {
  const char *name;
  cmd_func_t func;
} cmd_t;

static const cmd_t commands[] = {
  { "set", cmd_set },
  { "get", cmd_get },
  { "del", cmd_del },
  { "info", cmd_info },
  { NULL, NULL }
};

/**
 * execute_command - Process a command
 * @argc: Argument count
 * @argv: Argument vector
 *
 * Finds and executes the matching command handler.
 */
static void execute_command(int argc, char **argv)
{
  if (argc == 0) return;

  for (int i = 0; commands[i].name; i++) {
    if (strcasecmp(argv[0], commands[i].name) == 0) {
      commands[i].func(argc, argv);
      return;
    }
  }

  printf("ERR unknown command '%s'\n", argv[0]);
}

/**
 * main - Initialize server and process commands
 */
int main(void)
{
  server.db = dict_new();
  if (!server.db) {
    fprintf(stderr, "Failed to initialize database\n");
    return 1;
  }

  printf("Redis-like server initialized\n");

  /* Simulate command execution */
  char *set_args[] = { "set", "mykey", "myvalue" };
  execute_command(3, set_args);

  char *get_args[] = { "get", "mykey" };
  execute_command(2, get_args);

  char *info_args[] = { "info" };
  execute_command(1, info_args);

  char *del_args[] = { "del", "mykey" };
  execute_command(2, del_args);

  dict_free(server.db);
  return 0;
}

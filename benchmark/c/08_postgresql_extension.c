/*
 * custom_type_extension.c - PostgreSQL Custom Type Extension
 *
 * Implements a PostgreSQL extension with a custom type, functions,
 * and proper error handling using PostgreSQL's extended API.
 */

#include "postgres.h"
#include "fmgr.h"
#include "access/hash.h"
#include "utils/builtins.h"
#include "libpq-fe.h"
#include "executor/spi.h"

PG_MODULE_MAGIC;

/* Custom type: SimplePoint (x, y coordinates) */
typedef struct {
  int32 vl_len_;  /* Variable length header */
  float4 x;
  float4 y;
} SimplePoint;

#define SIMPLE_POINT_SIZE sizeof(SimplePoint)

/**
 * PG_FUNCTION_INFO_V1 - Register function with PostgreSQL
 *
 * This macro must precede every SQL-callable function.
 */

/* Input function: text -> SimplePoint */
PG_FUNCTION_INFO_V1(simplepoint_in);

Datum simplepoint_in(PG_FUNCTION_ARGS)
{
  char *str = PG_GETARG_CSTRING(0);
  SimplePoint *result;
  float4 x, y;
  int nargs;

  result = (SimplePoint *)palloc(SIMPLE_POINT_SIZE);
  SET_VARSIZE(result, SIMPLE_POINT_SIZE);

  nargs = sscanf(str, "(%f,%f)", &x, &y);

  if (nargs != 2) {
    ereport(ERROR,
            (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
             errmsg("invalid input syntax for type simplepoint: \"%s\"", str)));
  }

  result->x = x;
  result->y = y;

  PG_RETURN_POINTER(result);
}

/* Output function: SimplePoint -> text */
PG_FUNCTION_INFO_V1(simplepoint_out);

Datum simplepoint_out(PG_FUNCTION_ARGS)
{
  SimplePoint *point = (SimplePoint *)PG_GETARG_POINTER(0);
  char *result;

  result = palloc(64);
  snprintf(result, 64, "(%g,%g)", point->x, point->y);

  PG_RETURN_CSTRING(result);
}

/* Binary input function: bytea -> SimplePoint */
PG_FUNCTION_INFO_V1(simplepoint_recv);

Datum simplepoint_recv(PG_FUNCTION_ARGS)
{
  StringInfo buf = (StringInfo)PG_GETARG_POINTER(0);
  SimplePoint *result;

  result = (SimplePoint *)palloc(SIMPLE_POINT_SIZE);
  SET_VARSIZE(result, SIMPLE_POINT_SIZE);

  result->x = pq_getmsgfloat4(buf);
  result->y = pq_getmsgfloat4(buf);

  PG_RETURN_POINTER(result);
}

/* Binary output function: SimplePoint -> bytea */
PG_FUNCTION_INFO_V1(simplepoint_send);

Datum simplepoint_send(PG_FUNCTION_ARGS)
{
  SimplePoint *point = (SimplePoint *)PG_GETARG_POINTER(0);
  StringInfoData buf;

  pq_begintypsend(&buf);
  pq_sendfloat4(&buf, point->x);
  pq_sendfloat4(&buf, point->y);

  PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/* Equality operator */
PG_FUNCTION_INFO_V1(simplepoint_eq);

Datum simplepoint_eq(PG_FUNCTION_ARGS)
{
  SimplePoint *p1 = (SimplePoint *)PG_GETARG_POINTER(0);
  SimplePoint *p2 = (SimplePoint *)PG_GETARG_POINTER(1);

  if (p1->x == p2->x && p1->y == p2->y) {
    PG_RETURN_BOOL(true);
  } else {
    PG_RETURN_BOOL(false);
  }
}

/* Distance function */
PG_FUNCTION_INFO_V1(simplepoint_distance);

Datum simplepoint_distance(PG_FUNCTION_ARGS)
{
  SimplePoint *p1 = (SimplePoint *)PG_GETARG_POINTER(0);
  SimplePoint *p2 = (SimplePoint *)PG_GETARG_POINTER(1);
  float8 dx, dy, distance;

  dx = (float8)(p1->x - p2->x);
  dy = (float8)(p1->y - p2->y);
  distance = sqrt(dx * dx + dy * dy);

  PG_RETURN_FLOAT8(distance);
}

/* Hash function for use in hash indices */
PG_FUNCTION_INFO_V1(simplepoint_hash);

Datum simplepoint_hash(PG_FUNCTION_ARGS)
{
  SimplePoint *point = (SimplePoint *)PG_GETARG_POINTER(0);
  uint32 hash;

  hash = DatumGetUInt32(hash_uint32((uint32)point->x));
  hash ^= DatumGetUInt32(hash_uint32((uint32)point->y));

  PG_RETURN_UINT32(hash);
}

/* Comparison for sorting */
PG_FUNCTION_INFO_V1(simplepoint_cmp);

Datum simplepoint_cmp(PG_FUNCTION_ARGS)
{
  SimplePoint *p1 = (SimplePoint *)PG_GETARG_POINTER(0);
  SimplePoint *p2 = (SimplePoint *)PG_GETARG_POINTER(1);
  int cmp;

  if (p1->x < p2->x) {
    cmp = -1;
  } else if (p1->x > p2->x) {
    cmp = 1;
  } else if (p1->y < p2->y) {
    cmp = -1;
  } else if (p1->y > p2->y) {
    cmp = 1;
  } else {
    cmp = 0;
  }

  PG_RETURN_INT32(cmp);
}

/**
 * PG_FUNCTION_INFO_V1 - SQL aggregate function
 *
 * Demonstrates aggregate operations with SPI (Server Programming Interface).
 */

typedef struct {
  float8 sum_x;
  float8 sum_y;
  int count;
} PointAccum;

PG_FUNCTION_INFO_V1(simplepoint_accum);

Datum simplepoint_accum(PG_FUNCTION_ARGS)
{
  PointAccum *accum;
  SimplePoint *point;
  MemoryContext aggcontext, oldcontext;

  if (!AggCheckCallContext(fcinfo, &aggcontext)) {
    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
             errmsg("aggregate function must be called in aggregate context")));
  }

  if (PG_ARGISNULL(0)) {
    oldcontext = MemoryContextSwitchTo(aggcontext);
    accum = palloc0(sizeof(PointAccum));
    MemoryContextSwitchTo(oldcontext);
  } else {
    accum = (PointAccum *)PG_GETARG_POINTER(0);
  }

  if (!PG_ARGISNULL(1)) {
    point = (SimplePoint *)PG_GETARG_POINTER(1);
    accum->sum_x += point->x;
    accum->sum_y += point->y;
    accum->count++;
  }

  PG_RETURN_POINTER(accum);
}

PG_FUNCTION_INFO_V1(simplepoint_final);

Datum simplepoint_final(PG_FUNCTION_ARGS)
{
  PointAccum *accum;
  SimplePoint *result;

  if (PG_ARGISNULL(0)) {
    PG_RETURN_NULL();
  }

  accum = (PointAccum *)PG_GETARG_POINTER(0);

  if (accum->count == 0) {
    PG_RETURN_NULL();
  }

  result = (SimplePoint *)palloc(SIMPLE_POINT_SIZE);
  SET_VARSIZE(result, SIMPLE_POINT_SIZE);
  result->x = (float4)(accum->sum_x / accum->count);
  result->y = (float4)(accum->sum_y / accum->count);

  PG_RETURN_POINTER(result);
}

/**
 * Extension initialization
 *
 * Required function called by PostgreSQL when loading the extension.
 */

void _PG_init(void)
{
  /* Initialization code here */
}

void _PG_fini(void)
{
  /* Cleanup code here */
}

/**
 * Test utility function for SPI operations
 *
 * Demonstrates executing SQL queries from within a C extension.
 */

PG_FUNCTION_INFO_V1(simplepoint_query_example);

Datum simplepoint_query_example(PG_FUNCTION_ARGS)
{
  const char *sql = "SELECT COUNT(*) FROM pg_tables";
  int ret;
  uint64 count;

  ret = SPI_connect();
  if (ret != SPI_OK_CONNECT) {
    ereport(ERROR,
            (errcode(ERRCODE_INTERNAL_ERROR),
             errmsg("could not connect to SPI: %s", SPI_result_code_string(ret))));
  }

  ret = SPI_execute(sql, true, 0);
  if (ret != SPI_OK_SELECT) {
    SPI_finish();
    ereport(ERROR,
            (errcode(ERRCODE_INTERNAL_ERROR),
             errmsg("SPI_execute failed: %s", SPI_result_code_string(ret))));
  }

  if (SPI_processed > 0 && SPI_tuptable != NULL) {
    HeapTuple tuple = SPI_tuptable->vals[0];
    TupleDesc tupdesc = SPI_tuptable->tupdesc;
    bool isnull;

    count = DatumGetInt64(SPI_getbinval(tuple, tupdesc, 1, &isnull));
  } else {
    count = 0;
  }

  SPI_finish();

  PG_RETURN_INT64(count);
}

#include "holo_program.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 512U
#define OP_COUNT 10U
#define DATA_COUNT 3U
#define PROGRAM_LEN 4U

static void require(int ok, const char *msg)
{
    if (!ok) {
        fprintf(stderr, "test_program: %s\n", msg);
        exit(1);
    }
}

static void require_ok(int rc, const char *msg)
{
    if (rc != HOLO_OK) {
        fprintf(stderr, "test_program: %s: %s\n", msg, holo_strerror(rc));
        exit(1);
    }
}

static void add_into(double *acc, const double *row, size_t dim)
{
    for (size_t i = 0; i < dim; ++i) {
        acc[i] += row[i];
    }
}

static void make_instruction(holo_engine *engine,
                             const double *op_role,
                             const double *arg_role,
                             const double *op_vec,
                             const double *arg_vec,
                             double *out)
{
    double rows[2 * DIM];
    require_ok(holo_bind(engine, op_role, op_vec, rows), "bind op role");
    require_ok(holo_bind(engine, arg_role, arg_vec, rows + DIM), "bind arg role");
    require_ok(holo_bundle(DIM, rows, NULL, 2, out), "bundle instruction");
}

int main(void)
{
    holo_engine *engine = holo_engine_create(DIM, 123);
    double op_role[DIM];
    double arg_role[DIM];
    double positions[PROGRAM_LEN * DIM];
    double op_vectors[OP_COUNT * DIM];
    double op_norms[OP_COUNT];
    double data_vectors[DATA_COUNT * DIM];
    double data_norms[DATA_COUNT];
    double program[DIM] = {0.0};
    double instr[DIM];
    double bound[DIM];
    double expected_rows[2 * DIM];
    double expected[DIM];
    double out[DIM];
    int out_has_acc = 0;
    size_t trace_ops[PROGRAM_LEN];
    size_t trace_args[PROGRAM_LEN];
    size_t trace_count = 0;

    require(engine != NULL, "engine create");
    require_ok(holo_keygen_unitary(engine, 1, op_role), "op role");
    require_ok(holo_keygen_unitary(engine, 2, arg_role), "arg role");
    for (size_t i = 0; i < PROGRAM_LEN; ++i) {
        require_ok(holo_keygen_unitary(engine, 100 + i, positions + i * DIM), "position");
    }
    for (size_t i = 0; i < OP_COUNT; ++i) {
        require_ok(holo_keygen(engine, 200 + i, op_vectors + i * DIM), "opcode");
        op_norms[i] = holo_norm(DIM, op_vectors + i * DIM);
    }
    for (size_t i = 0; i < DATA_COUNT; ++i) {
        require_ok(holo_keygen(engine, 300 + i, data_vectors + i * DIM), "data");
        data_norms[i] = holo_norm(DIM, data_vectors + i * DIM);
    }

    make_instruction(engine,
                     op_role,
                     arg_role,
                     op_vectors + HOLO_PROGRAM_OP_LOAD * DIM,
                     data_vectors,
                     instr);
    require_ok(holo_bind(engine, positions, instr, bound), "program load");
    add_into(program, bound, DIM);

    make_instruction(engine,
                     op_role,
                     arg_role,
                     op_vectors + HOLO_PROGRAM_OP_BIND * DIM,
                     data_vectors + DIM,
                     instr);
    require_ok(holo_bind(engine, positions + DIM, instr, bound), "program bind");
    add_into(program, bound, DIM);

    make_instruction(engine,
                     op_role,
                     arg_role,
                     op_vectors + HOLO_PROGRAM_OP_BUNDLE * DIM,
                     data_vectors + 2 * DIM,
                     instr);
    require_ok(holo_bind(engine, positions + 2 * DIM, instr, bound), "program bundle");
    add_into(program, bound, DIM);

    make_instruction(engine,
                     op_role,
                     arg_role,
                     op_vectors + HOLO_PROGRAM_OP_HALT * DIM,
                     data_vectors,
                     instr);
    require_ok(holo_bind(engine, positions + 3 * DIM, instr, bound), "program halt");
    add_into(program, bound, DIM);
    require_ok(holo_normalize(DIM, program), "program normalize");

    require_ok(holo_bind(engine, data_vectors, data_vectors + DIM, expected_rows), "expected bind");
    memcpy(expected_rows + DIM, data_vectors + 2 * DIM, DIM * sizeof(expected_rows[0]));
    require_ok(holo_bundle(DIM, expected_rows, NULL, 2, expected), "expected bundle");

    require_ok(holo_program_run_basic(engine,
                                      program,
                                      positions,
                                      PROGRAM_LEN,
                                      op_role,
                                      arg_role,
                                      op_vectors,
                                      op_norms,
                                      OP_COUNT,
                                      data_vectors,
                                      data_norms,
                                      DATA_COUNT,
                                      NULL,
                                      0,
                                      PROGRAM_LEN,
                                      0.5,
                                      out,
                                      &out_has_acc,
                                      trace_ops,
                                      trace_args,
                                      PROGRAM_LEN,
                                      &trace_count),
               "program run basic");
    require(out_has_acc == 1, "runner produced accumulator");
    require(trace_count == 3, "trace count");
    require(trace_ops[0] == HOLO_PROGRAM_OP_LOAD && trace_args[0] == 0, "trace load");
    require(trace_ops[1] == HOLO_PROGRAM_OP_BIND && trace_args[1] == 1, "trace bind");
    require(trace_ops[2] == HOLO_PROGRAM_OP_BUNDLE && trace_args[2] == 2, "trace bundle");
    require(holo_cosine(DIM, out, expected) > 0.999, "program accumulator matches expected");

    holo_engine_destroy(engine);
    puts("test_program ok");
    return 0;
}

// Compile against the installed QUactlize checkout; no provider code is copied.
#include "quactlize/dense/api.h"
#include <cstddef>
static_assert(sizeof(quactlize_ppu_placed_arrangement_v2) == 40);
static_assert(sizeof(quactlize_ppu_kpack_sizes_v1) == 32);
static_assert(sizeof(quactlize_ppu_dense_options_v1) == 16);
static_assert(sizeof(quactlize_ppu_dense_params_v1) == 112);
static_assert(offsetof(quactlize_ppu_dense_params_v1, arrangement) == 40);
static_assert(offsetof(quactlize_ppu_dense_params_v1, sizes) == 80);
static_assert(sizeof(quactlize_ppu_dense_plan_info_v1) == 224);
static_assert(offsetof(quactlize_ppu_dense_plan_info_v1, config) == 32);

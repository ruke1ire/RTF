// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_EXECUTION_OPERATORS_HPP
#define CUFFTDX_OPERATORS_EXECUTION_OPERATORS_HPP

#include "../detail/expressions.hpp"

namespace cufftdx {
    struct Thread: detail::operator_expression {};
    struct Block: detail::operator_expression {};
} // namespace cufftdx

#endif // CUFFTDX_OPERATORS_EXECUTION_OPERATORS_HPP

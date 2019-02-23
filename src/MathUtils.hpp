#pragma once

namespace tfutils
{

template<typename T>
T map(const T value, const T inputMin, const T inputMax, const T outputMin, const T outputMax)
{
    return ((value - inputMin) / (inputMax - inputMin) * (outputMax - outputMin) + outputMin);
}

} // namespace tfutils
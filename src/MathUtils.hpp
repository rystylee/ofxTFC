#pragma once

namespace tfutils
{

    template<typename T>
    void map(std::vector<std::vector<T>>& values, const T inputMin, const T inputMax, const T outputMin, const T outputMax)
    {
        for (auto& batch : values)
        {
            map(batch, inputMin, inputMax, outputMin, outputMax);
        }
    }

    template<typename T>
    void map(std::vector<T>& values, const T inputMin, const T inputMax, const T outputMin, const T outputMax)
    {
        for (auto& v : values)
        {
            v = map(v, inputMin, inputMax, outputMin, outputMax);
        }
    }

    template<typename T>
    T map(const T value, const T inputMin, const T inputMax, const T outputMin, const T outputMax)
    {
        return ((value - inputMin) / (inputMax - inputMin) * (outputMax - outputMin) + outputMin);
    }

} // namespace tfutils
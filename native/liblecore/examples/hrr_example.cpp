#include <lecore/lecore.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

struct ContextDeleter {
    void operator()(lecore_context *context) const noexcept
    {
        lecore_context_destroy(context);
    }
};

using Context = std::unique_ptr<lecore_context, ContextDeleter>;

int main()
{
    const std::array<double, 4> role{1.0, 2.0, 0.0, -1.0};
    const std::array<double, 4> value{2.0, 0.0, 1.0, 0.0};
    std::array<double, 4> composite{};
    lecore_config_v0 config{};
    lecore_context *raw_context = nullptr;

    lecore_config_init_v0(&config);
    config.dimension = static_cast<std::uint32_t>(role.size());
    const lecore_status create_status =
        lecore_context_create(&config, &raw_context);
    if (create_status != LECORE_OK) {
        throw std::runtime_error(lecore_status_string(create_status));
    }
    Context context{raw_context};

    const lecore_status bind_status = lecore_hrr_bind_f64(
        context.get(), role.data(), value.data(), composite.data());
    if (bind_status != LECORE_OK) {
        throw std::runtime_error(lecore_status_string(bind_status));
    }

    for (const double component : composite) {
        std::cout << component << ' ';
    }
    std::cout << '\n';
    return 0;
}

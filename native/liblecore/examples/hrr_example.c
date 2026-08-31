#include <lecore/lecore.h>

#include <stdio.h>

int main(void)
{
    const double role[4] = {1.0, 2.0, 0.0, -1.0};
    const double value[4] = {2.0, 0.0, 1.0, 0.0};
    double composite[4];
    lecore_config_v0 config;
    lecore_context *context = NULL;
    lecore_status status;
    size_t index;

    lecore_config_init_v0(&config);
    config.dimension = 4;
    status = lecore_context_create(&config, &context);
    if (status != LECORE_OK) {
        fprintf(stderr, "context: %s\n", lecore_status_string(status));
        return 1;
    }

    status = lecore_hrr_bind_f64(context, role, value, composite);
    if (status != LECORE_OK) {
        fprintf(stderr, "bind: %s\n", lecore_status_string(status));
        lecore_context_destroy(context);
        return 1;
    }

    for (index = 0; index < 4; ++index) {
        printf("%s%.6f", index == 0 ? "" : " ", composite[index]);
    }
    putchar('\n');
    lecore_context_destroy(context);
    return 0;
}

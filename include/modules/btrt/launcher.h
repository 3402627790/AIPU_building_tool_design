#ifndef __LAUNCHER_H__
#define __LAUNCHER_H__

#include <stdint.h>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

namespace aipubt::rt
{

typedef struct tensor_data
{
    void *data;             // if it is an input request, data should be initialized
    uint32_t length;        // actual data length being used
    std::string name;      // input tensor name
    std::vector<uint32_t> shape; // When using dynamic shapes, each frame of input data must have a shape.
} tensor_data_t;

// Callback function for asynchronized execution
// int&  : callback retrun value
//  std::vector<tensor_data_t> : output datas
// void*  ： callback function context,
typedef std::function<void(int &, std::vector<tensor_data_t>, void *)> AsyncCallback;

class LauncherImpl;

class Launcher
{
public:
    Launcher();
    ~Launcher();
    int load_from_mem(const void *elf_data, size_t size);

    int load_from_file(const std::string &elf_filepath);

    int run(const std::vector<tensor_data_t> &inputs = {},
            const std::map<std::string, int> &symbols = {});

    int run_async(AsyncCallback callback, const std::vector<tensor_data_t> &inputs = {},
                  const std::map<std::string, int> &symbols = {});

    std::vector<tensor_data_t> get_inputs();

    std::vector<tensor_data_t> get_outputs();

    int dump_outputs(const std::map<int, std::string> &tensor_names = {});

    int bind_device(int device_id);

    int set_run_priority(uint8_t priority);

    uint8_t get_run_priority();

private:
    std::unique_ptr<LauncherImpl> impl_;
};

} // namespace aipubt::rt

#endif //!__LAUNCHER_H__
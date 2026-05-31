#include "asmtranscode.h"
#include "utils/os_utils.h"
#include <fstream>
namespace utils
{
std::string exec(const char *cmd);

}
namespace aipubt
{

std::map<uint32_t, CGFunc> &get_trans_map();

const std::map<uint32_t, CGFunc> &TransCode::trans_map()
{
    return get_trans_map();
}
std::string TransCode::dis(const std::string &path)
{
    std::string p = path;
    if (path.empty())
    {
        p = "tmp.text";
    }
    std::ofstream of(p, std::ios::binary);
    std::vector<std::pair<uint32_t, CodeMap::CodeGenPtr>> codes(m_codemap.m_ptr->cgs.begin(), m_codemap.m_ptr->cgs.end());
    std::sort(codes.begin(), codes.end());
    for (auto &code : codes)
    {
        of.write((char *)code.second->getCode(), code.second->getSize());
    }
    of.close();
    std::string cmd = "objdump -D -Mintel,x86-64 -b binary -m i386 " + p;
    return utils::exec(("objdump -D -Mintel,x86-64 -b binary -m i386 " + p).c_str()).substr(84);
}

void TransCode::disassembly(const uint8_t *d, uint32_t size)
{
    std::ofstream of("tmp.text");
    of.write((char *)d, size);
    of.close();
    std::string cmd = "objdump -D -Mintel,x86-64 -b binary -m i386 tmp.text";
    std::cout << utils::exec("objdump -D -Mintel,x86-64 -b binary -m i386 tmp.text").substr(84);
}

void TransCode::disassembly(uint32_t offset)
{
    disassembly(get_code<uint8_t *>(offset), get_size(offset));
}
TransCodeRegisterHelper::TransCodeRegisterHelper(InstructionType t, CGFunc f)
{
    get_trans_map()[(uint32_t)t] = f;
}
}; // namespace aipubt
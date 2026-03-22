//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang 
 * @Date: 2020-01-29 14:28:55 
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-05-31 12:02:47
 */
#ifndef NEO_ARGPARSER_H
#define NEO_ARGPARSER_H

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#define USE_BT_ANY

#ifdef USE_BT_ANY
#include <core/any.h>
using namespace aipubt;
#endif
using namespace std;
class ArgParser
{
private:
#ifdef USE_BT_ANY
    using Any = aipubt::Any;
#else
    class Any;
#endif
    struct Option;
    map<string, int> m_index;
    map<string, int> m_pos_index;
    string m_program_name;
    vector<Option *> m_options;
    vector<Option *> m_position_options;
    set<Option *> m_hidden_options;
    map<string, Option *> m_args;
    bool is_check = false;

    void check_args()
    {
        if (is_check)
        {
            return;
        }
        is_check = true;
        ostringstream oss;
        for (auto opt : m_position_options)
        {
            if (!m_args.count(opt->name))
            {
                oss << opt->name << " ";
            }
        }
        auto msg = oss.str();
        if (!msg.empty())
        {
            parse_error("Required position arguments: " + msg, true);
        }
    }
    void add_position_options(Option *opt)
    {
        m_position_options.push_back(opt);
    }
    void parse_error(const string &msg, bool show_usage = false, bool show_hidden = false)
    {
        std::cerr << "ArgParser Error: " << msg << std::endl;
        if (show_usage)
            std::cerr << usage(show_hidden) << std::endl;
        exit(-5);
    }
    bool is_position_arg(const string &name)
    {
        return name[0] != '-';
    }
    string verify(const string &name)
    {
        return name;
    }
    static string strip(const string &name)
    {
        size_t begin = 0;
        begin += name.size() > 0 ? name[0] == '-' : 0;
        begin += name.size() > 3 ? name[1] == '-' : 0;
        return name.substr(begin);
    }
    static string upper(const string &in)
    {
        string out(in);
        std::transform(out.begin(), out.end(), out.begin(), ::toupper);
        return out;
    }
    inline bool has_only_digits(const string &s)
    {
        return s.find_first_not_of("0123456789") == string::npos;
    }
    inline bool is_hex_number(const string &s)
    {
        return s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') && s.substr(2).find_first_not_of("0123456789abcedfABCDEF") == string::npos;
    }
    inline bool is_number(const std::string &s)
    {
        std::istringstream iss(s);
        double f;
        iss >> std::noskipws >> f; // noskipws considers leading whitespace invalid
        // Check the entire string was consumed and if either failbit or badbit is set
        return iss.eof() && !iss.fail();
    }

public:
    ArgParser(){};
    ~ArgParser()
    {
        for (auto opt : m_options)
        {
            delete opt;
        }
        for (auto opt : m_position_options)
        {
            delete opt;
        }
    };
    void add_argument(const string &name, const string &help)
    {
        add_argument("", name, help, true, false);
    }
    void add_hidden_argument(const string &name, const string &help)
    {
        add_hidden_argument("", name, help, true, false);
    }
    template <typename T>
    void add_argument(const string &name, const string &help, const T &default_value)
    {
        add_argument("", name, help, true, false, default_value, true);
    }
    template <typename T>
    void add_hidden_argument(const string &name, const string &help, const T &default_value)
    {
        add_argument("", name, help, true, false, default_value, true, true);
    }

    template <typename T>
    void add_argument(const string &_short_name, const string &name, const string &help, const T &default_value)
    {
        add_argument(_short_name, name, help, true, false, default_value, true);
    }
    template <typename T>
    void add_hidden_argument(const string &_short_name, const string &name, const string &help, const T &default_value)
    {
        add_argument(_short_name, name, help, true, false, default_value, true, true);
    }

    void add_argument(const string &_short_name, const string &_name, const string &help, bool optional, bool store_true)
    {
        add_argument(_short_name, _name, help, optional, store_true, 0);
    }
    void add_hidden_argument(const string &_short_name, const string &_name, const string &help, bool optional, bool store_true)
    {
        add_argument(_short_name, _name, help, optional, store_true, 0, false, true);
    }

    template <typename T>
    void add_hidden_argument(const string &_short_name, const string &_name,
                             const string &help, bool optional,
                             bool store_true, const T &default_value, bool has_default_value = false)
    {
        add_argument(_short_name, _name, help, optional,
                     store_true, default_value, has_default_value, true);
    }
    template <typename T>
    void add_argument(const string &_short_name, const string &_name,
                      const string &help, bool optional,
                      bool store_true, const T &default_value, bool has_default_value = false, bool is_hidden = false)
    {
        auto short_name = verify(_short_name);
        auto name = verify(_name);
        auto opt = new Option(short_name, name, help, default_value, optional, store_true);
        opt->has_default = has_default_value;
        if (is_hidden)
        {
            m_hidden_options.insert(opt);
        }
        auto pos = is_position_arg(name);
        map<string, int> *index = nullptr;
        vector<Option *> *options = nullptr;
        if (pos)
        {
            options = &m_position_options;
            index = &m_pos_index;
        }
        else
        {
            options = &m_options;
            index = &m_index;
        }
        options->push_back(opt);
        if (!short_name.empty())
        {
            (*index)[short_name] = options->size() - 1;
            (*index)[strip(short_name)] = options->size() - 1;
        }
        if (!name.empty())
        {
            (*index)[name] = options->size() - 1;
            (*index)[strip(name)] = options->size() - 1;
        }
        if (has_default_value)
        {
            m_args[strip(opt->name)] = opt;
        }
    }
    void parse(int argc, const char **argv)
    {
        parse(vector<string>(argv, argv + argc));
    }
    void parse(const vector<string> &argv)
    {
        m_program_name = argv[0];
        int position_arg_count = m_position_options.size();
        for (unsigned int i = 1; i < argv.size(); ++i)
        {
            string name, value;
            Option *opt = nullptr;
            auto arg = argv[i];
            auto idx = arg.find("=");
            if (idx != string::npos)
            {
                name = arg.substr(0, idx);
                value = arg.substr(idx + 1);
            }
            else
            {
                name = arg;
            }
            if (!m_index.count(name))
            {
                if (!name.empty() && name[0] != '-' && position_arg_count > 0)
                {
                    opt = m_position_options[m_position_options.size() - position_arg_count];
                    opt->value = name;
                    position_arg_count--;
                    m_args[strip(opt->name)] = opt;
                    continue;
                }
                else
                {
                    parse_error("Unknown option:" + arg, true);
                }
            }
            else
            {
                opt = m_options[m_index[name]];
            }

            if (!opt->store_true && value.empty())
            {
                if (argv.size() <= i + 1 || (argv[i + 1][0] == '-' && m_index.count(argv[i + 1])))
                {
                    bool show_hidden = m_hidden_options.count(opt);
                    parse_error("option:" + opt->name + " requires an argument value. Help info:\n\t" + opt->help + "\n", true, show_hidden);
                    return;
                }
                value = argv[i + 1];

                i++;
            }
            std::istringstream iss(value);
            if (has_only_digits(value))
            {
                int64_t v;
                iss >> v;
                opt->value = v;
            }
            else if (is_hex_number(value))
            {
                uint64_t v;
                iss >> std::hex >> v;
                opt->value = v;
            }
            else if (is_number(value))
            {
                float v = std::atof(value.c_str());
                opt->value = v;
            }
            else if (value == "True" || value == "true" || value == "TRUE")
            {
                opt->value = true;
            }
            else if (value == "False" || value == "false" || value == "FALSE")
            {
                opt->value = false;
            }
            else
            {
                opt->value = value;
            }
            m_args[strip(opt->name)] = opt;
        }
    }
    template <typename T>
    T &get(const string &key)
    {
        check_args();
        if (!m_args.count(key))
        {
            parse_error("Unkown key:" + key);
        }
        return cast_any<T>(m_args[key]->value);
    }
    bool exists(const string &key)
    {
        return m_args.count(key);
    }

    // return all non-position arguments, i.e. options
    std::vector<std::string> get_options()
    {
        std::vector<std::string> ret;
        for (auto arg : m_args)
        {
            for (auto opt : m_position_options)
            {
                if (arg.first == opt->name)
                {
                    continue;
                }
            }
            ret.push_back(arg.first);
        }
        return ret;
    }
    string full_usage()
    {
        return usage(true);
    }
    string usage(bool show_hidden = false)
    {
        ostringstream oss;
        oss << m_program_name << "\nUsage:\n";
        if (m_position_options.size())
        {
            oss << "positional arguments:\n";
            for (auto opt : m_position_options)
            {
                if (m_hidden_options.count(opt) && !show_hidden)
                {
                    continue;
                }
                oss << "  " << opt->name << "\n  \t";
                oss << opt->help << "\n";
            }
        }
        if (m_options.size())
        {
            oss << "optional arguments:\n";
        }
        for (auto opt : m_options)
        {
            if (m_hidden_options.count(opt) && !show_hidden)
            {
                continue;
            }
            oss << "  ";
            if (opt->store_true)
            {
                if (!opt->short_name.empty())
                    oss << opt->short_name << ",";
                oss << opt->name << "\n  \t";
            }
            else
            {
                if (!opt->short_name.empty())
                {
                    oss << opt->short_name << " <" << upper(strip(opt->name)) << ">,";
                }
                oss << opt->name << " <" << upper(strip(opt->name)) << ">\n  \t";
            }
            oss << opt->help << "\n";
        }
        return oss.str();
    }

    const string &program_name()
    {
        return m_program_name;
    }
    std::map<std::string, Option *> get_args()
    {
        return m_args;
    }

private:
#ifndef USE_BT_ANY
    class Any
    {
    public:
        class BaseData
        {
        public:
            virtual const std::type_info &get_type() = 0;
            virtual BaseData *clone() = 0;
            virtual ~BaseData(){};
        };
        template <typename T>
        class MData : public BaseData
        {
        public:
            MData(const T &data) : m_data(data) // this->m_data=data
            {
            }
            ~MData() override {}

            const std::type_info &get_type() override
            {
                return typeid(m_data);
            }

            T &get_value()
            {
                return m_data;
            }

            BaseData *clone() override
            {
                return new MData(m_data);
            }

        private:
            T m_data;
        };

        inline Any() = default;
        Any(const Any &other)
        {
            if (m_value)
                delete m_value;
            m_value = other.m_value->clone();
        }
        template <typename T>
        Any(const T &x)
        {
            m_value = new MData<T>(x);
        }
        ~Any()
        {
            if (m_value)
            {
                delete m_value;
                m_value = nullptr;
            }
        }

        inline BaseData *value() const
        {
            return m_value;
        }

        Any &operator=(const Any &other)
        {
            if (m_value)
                delete m_value;
            m_value = other.m_value->clone(); //TODO
            return *this;
        }

        template <typename T>
        Any &operator=(const T &other)
        {
            if (m_value)
                delete m_value;
            m_value = new MData<T>(other);
            return *this;
        }

    private:
        BaseData *m_value = nullptr;
    };

    template <typename T>
    T &cast_any(const Any &x)
    {
        Any::BaseData *ptr = x.value();
        return static_cast<Any::MData<T> *>(ptr)->get_value();
    }

#endif
    struct Option
    {
        string short_name;
        string name;
        string help;
        Any value;
        bool optional;
        bool store_true;
        bool has_default;
        Option(){};
        Option(const string &short_name, const string &name,
               const string &help,
               bool optional = true,
               bool store_true = false)
            : short_name(short_name), name(name), help(help), optional(optional), store_true(store_true)
        {
            value = !store_true;
            has_default = false;
        }
        Option(const string &short_name, const string &name,
               const string &help, const Any &default_value, bool optional = true,
               bool store_true = false)
            : short_name(short_name), name(name), help(help), value(default_value), optional(optional), store_true(store_true)
        {
            if (store_true)
            {
                value = false;
            }
            has_default = true;
        }
    };
};
#endif //!NEO_ARGPARSER_H
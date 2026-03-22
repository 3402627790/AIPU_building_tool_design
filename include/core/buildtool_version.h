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
#ifndef BUILDTOOL_TOOLS_VERSION_H
#define BUILDTOOL_TOOLS_VERSION_H
#include <stdint.h>
#include <string>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#ifndef MAJOR_VERSION
#define MAJOR_VERSION 6
#endif

#ifndef MINOR_VERSION
#define MINOR_VERSION 1
#endif

#ifndef BUILDTOOL_BUILD_NUMBER
#define BUILDTOOL_BUILD_NUMBER 1
#endif //!BUILDTOOL_BUILD_NUMBER

#define VERSION_NUMBER ((((MAJOR_VERSION)&0xff) << 24) | (((MINOR_VERSION)&0xff) << 16) | (BUILDTOOL_BUILD_NUMBER & 0xffff))
#define AIPUBIN_BT_VERSION_NUMBER ((((MAJOR_VERSION)&0xff)) | (((MINOR_VERSION)&0xff) << 8) | (BUILDTOOL_BUILD_NUMBER & 0xffff) << 16)
#define VERSION (STR(MAJOR_VERSION) "." STR(MINOR_VERSION) "." STR(BUILDTOOL_BUILD_NUMBER))
#ifndef GIT_COMMIT_ID
#define GIT_COMMIT_ID ""
#endif
struct Versioning
{
    uint8_t major = MAJOR_VERSION;
    uint8_t minor = MINOR_VERSION;
    uint16_t patch = BUILDTOOL_BUILD_NUMBER;
    std::string version = VERSION;
    std::string commit = GIT_COMMIT_ID;
    static const Versioning &get_version()
    {
        static Versioning g;
        return g;
    }
};

#define DECLARE_VERSION(modulename) \
    Versioning modulename##_version = {MAJOR_VERSION, MINOR_VERSION, BUILDTOOL_BUILD_NUMBER, VERSION, GIT_COMMIT_ID};
#endif //!BUILDTOOL_TOOLS_VERSION_H
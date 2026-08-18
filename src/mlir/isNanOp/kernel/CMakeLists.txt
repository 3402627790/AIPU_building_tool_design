set(MODULE_NAME "isnan")
set(MODULE_DIR ${KERNEL_ROOT_DIR}/${MODULE_NAME})

register_kernel_objects(
    #### must have: MODULE_NAME ####
    MODULE_NAME "${MODULE_NAME}"

    #### shoule have: KERNEL_FILE or ASM_FILES ####
    KERNEL_FILE "${MODULE_DIR}/${MODULE_NAME}_kernel.cps"
    # ASM_FILES "${MODULE_DIR}/${MODULE_NAME}_asm_float32_int32.s"
    # ASM_FILES 
    #     "${MODULE_DIR}/${MODULE_NAME}_asm_float32_int32.s"
    #     "${MODULE_DIR}/${MODULE_NAME}_asm_float16_int16.s"
    #     "${MODULE_DIR}/${MODULE_NAME}_asm_bfloat16_int16.s"

    #### optional ####
    # KERNEL_COMPILE_FLAGS "-xxx;-zzz" ### if you want to add additional cpscc flags
)
# Profiler

The doc would propose a Profiler design based on current software stack. 

The profiler is designed to be a layer-level profiling tool. The layer (operator) is the basic profiling object. More fine-grain profiler would not be discussed here, it may be designed as a part of C-compiler.

For the layer-level profiler, it is better to integrate into GBuilder, which is a layer-level linker. 

Below we will discus how a profiler works:

### Normal execution V.S profiler execution

The following figure shows the normal execution and profiling execution.

In normal execution, at the beginning of the code, a `init_ctx` code would initialize the running context, it will initialize the stack pointer, all R register and the Parameter Pointer etc.

After  init_ctx, the first layer would be run. All layers can be either scalar/TPC code, either scalar/HWA code or scalar/TPC/HWA mixed code.

After a layer done, a piece of code named `layer connector` would update the parameter pointer, and update other context variables, then followed by other layer.

After all layers done, a `exit` code would clean the cache and exit.

An interrupt handler should be implemented for profiling, because in ZYQian architecture, the AIFF(AI Fixed Function, HWA in ZYv1) descriptor can be a linked list, which means that all in order AIFF layers can be merged in one linked list. So, between two layer, there is no scalar code. So we need add an interrupt handler for the `AIFF done` interrupt.

![](images/profiler/profile_vs_normal.svg)

Below is how AIFF interrupt works:
1. we send a start command from scalar to AIFF(HWA)
2. the HWA would run descriptor, meanwhile, the scalar or TPC can do other work, or not.
3. Once AIFF(HWA) done, a `AIFF(HWA) done` interrupt will be triggered.
4. The normal execution would be hold, and the interrupt handler start to handler the interrupt.
5. One interrupt handler finishes its work, it will return to normal execution.
6. AIFF(HWA) would execute the new descriptor(if is a linked descriptor), and scalar would run other code.


![](images/profiler/hwa_descriptor_interrupt.svg)


For profiling each layer, we can add some `profiling code` before and after a layer as the figure shown. In the `profiling code` we just need to do:

1. save the running context
2. read performance data from performance counter
3. save the performance data to a specific memory buffer.
4. restore the context, and return the controller to layer op.

![](images/profiler/profile_code.svg)

For interrupt, we need check the interrupt code first, if is the `AIFF done` or other profiling interrupt, then we jump to profiling code to record the performance data then return to normal execution, else we transfer the controller to other handler. 


According to the Hardware document, we have 4  groups of performance-counter-register, which means we can count 4 types of event. Below is a table for all available events for performance counter for ZYv1. We need select 4 of these event, and setup the 4 group performance counter register after the `init_ctx`, then at the end of the execution of each layer, the profiling code can read the counter registers to get our profiling data.


| Event number | Event                                    | Description                                                                         |   |   |
|--------------|------------------------------------------|-------------------------------------------------------------------------------------|---|---|
| **0**           | **Cycle**                                    | **Count cycle.**                                                                      |   |   |
| **1**            | **VLIW package**                             | **Count package.**                                                                      |   |   |
| 2            | Instruction                              | Count all instructions except NOP.                                                  |   |   |
| 3            | Slot 0 in instruction package valid      | Count instruction in slot0.                                                         |   |   |
| 4            | Slot 1 in instruction package valid      | Count instruction in slot1.                                                         |   |   |
| 5            | Slot 2 in instruction package valid      | Count instruction in slot2.                                                         |   |   |
| 6            | Slot 3 in instruction package valid      | Count instruction in slot3.                                                         |   |   |
| 7            | Scalar instruction                       | Count scalar instruction except NOP.                                                |   |   |
| 8            | Vector instruction                       | Count vector instruction except NOP.                                                |   |   |
| **9**            | **HWA instruction**                          | **Count instruction sent to HWA.**                                                      |   |   |
| 10           | Branch                                   | Count branch instruction.                                                           |   |   |
| 11           | Branch miss                              | Count branch mis-prediction.                                                        |   |   |
| 12           | Instruction cache miss                   | Count Icache miss.                                                                  |   |   |
| 13           | Instruction cache hit                    | Count Icache hit.                                                                   |   |   |
| 14           | Icache way-prediction miss               | Count Icache way-prediction miss.                                                   |   |   |
| 15           | Level-one data cache miss                | Cacheable load or store miss, only for the first access to Dcache.                  |   |   |
| 16           | Level-one data cache hit                 | Cacheable load or store hit, only for the first access to dcache                    |   |   |
| 17           | Level-two cache miss                     | (L2 cache is not implemented).                                                      |   |   |
| 18           | Level-two cache hit                      | (L2 cache is not implemented).                                                      |   |   |
| 19           | Scalar load                              | All load instructions of scalar, including cacheable and others, including pop.     |   |   |
| 20           | Scalar store                             | All store instructions of scalar, including cacheable and others, including push.   |   |   |
| 21           | Scalar LSQ full                          | Count cycles of full state.                                                         |   |   |
| 22           | Scalar LSQ WAW hazard                    | Count load to the same GPR issue.                                                   |   |   |
| 23           | Scalar LSQ memory hazard                 | Count Store-load or Store-store conflict.                                           |   |   |
| 24           | Scalar LSQ memory bypass                 | Count Store-load bypass.                                                            |   |   |
| 25           | Scalar MQ full                           | Count cycles of full state.                                                         |   |   |
| 26           | Scalar CLQ full                          | Count cycles of full state.                                                         |   |   |
| 27           | Scalar WBQ full                          | Count cycles of full state.                                                         |   |   |
| 28           | Scalar cache flush                       | Count COP instruction.                                                              |   |   |
| 29           | Scalar cache prefetch miss               | Count cache lines being prefetched by CPM.                                          |   |   |
| 30           | Scalar cache prefetch hit and cancelled  | Count cache lines prefetches being cancelled by HW due to L1D, MQ or CLQ hit.       |   |   |
| 31           | Scalar L1D dirty way replacement         | Count Dirty way replacement to L2.                                                  |   |   |
| 32           | Scalar L1D2L2 address channel busy       | Count waiting cycle.                                                                |   |   |
| 33           | Scalar L1D2L2 data channel busy          | Count waiting cycle.                                                                |   |   |
| -            | -                                        | -                                                                                   |   |   |
| 48           | GSRAM credit empty                       | Count credit busy cycle of GSRAM direction.                                         |   |   |
| 49           | GSRAM Vector load                        | Count vector load instructions that really access GSRAM.                            |   |   |
| 50           | GSRAM Vector store                       | Count vector store instructions that really access GSRAM.                           |   |   |
| 51           | GSRAM Scalar load                        | Count scalar load instructions that really access GSRAM.                            |   |   |
| 52           | GSRAM Scalar store                       | Count scalar store instructions that really access GSRAM.                           |   |   |
| 53           | GSRAM conflict for TPC                   | Count waiting cycles of GSRAM bank conflict when TPC access.                        |   |   |
| 54           | GSRAM conflict for DMA                   | Count waiting cycles of GSRAM when DMA and TPC both request GSRAM.                  |   |   |
| 55           | GSRAM LSQ full                           | Count the cycles that GSRAM being full.                                             |   |   |
| -            | -                                        | -                                                                                   |   |   |
| 64           | TPC credit empty                         | Count credit busy cycle of TPC direction.                                           |   |   |
| 65           | TPC issue queue full                     | Count TPC issue queue full cycles.                                                  |   |   |
| 66           | TPC hazard on gsram busy                 | Count TPC and GSRAM request busy cycles.                                            |   |   |
| 67           | TPC hazard on scalar wb port             | Count TPC write scalar RF busy cycles.                                              |   |   |
| 68           | TPC slot1 source 3 foR/Ward              | Count TPC slot1 internal foR/Ward times.                                            |   |   |
| 69           | TPC issue hazard: RAW                    | Count TPC’s RAW hazard cycles.                                                      |   |   |
| 70           | TPC issue hazard: WAW                    | Count TPC’s WAW hazard cycles.                                                      |   |   |
| 71           | TPC issue hazard: LSQ0                   | Count TPC’s SQ0 full hazard cycles.                                                 |   |   |
| 72           | TPC issue hazard: LSQ1                   | Count TPC’s LSQ1 full hazard cycles.                                                |   |   |
| 73           | TPC issue hazard: scalar                 | Count TPC’s scalar data read hazard cycles.                                         |   |   |
| -            | -                                        | -                                                                                   |   |   |
| 81           | HWA: execute times                       | Count HWA execute times, including instruction enable HWA and crb enable HWA.       |   |   |
| 82           | HWA: busy cycle                          | Count the cycles when HWA is busy.                                                  |   |   |
| 83           | HWA: dbrg busy cycle                     | Count the cycles when dbrg is busy.                                                 |   |   |
| 84           | HWA: read request to dma                 | Count the times of dbrg read request to DMA.                                        |   |   |
| 85           | HWA: write request to dma                | Count the times of dbrg write request to DMA.                                       |   |   |
| 86           | HWA: sys read config file request        | Count the times of sys read config file request to dbrg.                            |   |   |
| 87           | HWA: conv read weight request            | Count the times of conv read weight request to dbrg.                                |   |   |
| 88           | HWA: conv read data request              | Count the times of conv read data request to dbrg.                                  |   |   |
| 89           | HWA: sdp read request                    | Count the times of sdp read request to dbrg.                                        |   |   |
| 90           | HWA: tdp write request                   | Count the times of tdp write request to dbrg.                                       |   |   |
| **91**           | **HWA: PE cnosuming**                        | **Count the cycles when PE is busy.**                                                   |   |   |
| -            | -                                        | -                                                                                   |   |   |
| 108          | DMA: AXI read request sent               | Count DMA read request triggered by SPU.                                            |   |   |
| 109          | DMA: AXI read request burst length       | Count DMA read request length of SPU, used to calculate the average length of read. |   |   |
| 110          | DMA: AXI read request outstanding number | Count the number of current outstanding DMA read request, multi-bit format.         |   |   |
| 111          | DMA: AXI write request sent              | Delay from DMA send write request to write response back, under the command SPU.    |   |   |
| 112          | DMA: AXI write request burst length      | Burst length of DMA send write request, under the command of SPU.                   |   |   |
| -            | -                                        | -                                                                                   |   |   |
| 118          | MIF: AXI read request valid              | Count the AIPU valid read request number                                            |   |   |
| 119          | MIF: AXI full size read request valid    | Count the AIPU full size valid read request number                                  |   |   |
| 120          | MIF: AXI read data valid                 | Count the AIPU valid read data number                                               |   |   |
| 121          | MIF: AXI write request valid             | Count the AIPU valid write request number                                           |   |   |
| 122          | MIF: AXI full size write request valid   | Count the AIPU full size valid write request number                                 |   |   |
| 123          | MIF: AXI write data valid                | Count the AIPU valid write data number                                              |   |   |
| -            | -                                        | -                                                                                   |   |   |
| 128          | Tpc_tec0_ls0: lsq full                   | Local load/store queue of ls0 is full.                                              |   |   |
| 129          | Tpc_tec0_ls0: gsq full                   | Local gather/scatter queue of ls0 is full.                                          |   |   |
| 130          | Tpc_tec0_ls0: sdq full                   | Local store data queue of ls0 is full.                                              |   |   |
| 131          | Tpc_tec0_ls0: load valid                 | Count load executed in ls0.                                                         |   |   |
| 132          | Tpc_tec0_ls0: store valid                | Count store executed in ls0.                                                        |   |   |
| 133          | Tpc_tec0_ls1: lsq full                   | Local load/store queue of ls1 is full.                                              |   |   |
| 134          | Tpc_tec0_ls1: load valid                 | Count load executed in ls1.                                                         |   |   |
| 135          | Tpc_tec0_ls1: store valid                | Count store executed in ls1.                                                        |   |   |
| 136          | Tpc_tec1_ls0: lsq full                   | Local load/store queue of ls0 is full.                                              |   |   |
| 137          | Tpc_tec1_ls0: gsq full                   | Local gather/scatter queue of ls0 is full.                                          |   |   |
| 138          | Tpc_tec1_ls0: sdq full                   | Local store data queue of ls0 is full.                                              |   |   |
| 139          | Tpc_tec1_ls0: load valid                 | Count load executed in ls0.                                                         |   |   |
| 140          | Tpc_tec1_ls0: store valid                | Count store executed in ls0.                                                        |   |   |
| 141          | Tpc_tec1_ls1: lsq full                   | Local load/store queue of ls1 is full.                                              |   |   |
| 142          | Tpc_tec1_ls1: load valid                 | Count load executed in ls1.                                                         |   |   |
| 143          | Tpc_tec1_ls1: store valid                | Count store executed in ls1.                                                        |   |   |
| 144          | Tpc_tec2_ls0: lsq full                   | Local load/store queue of ls0 is full.                                              |   |   |
| 145          | Tpc_tec2_ls0: gsq full                   | Local gather/scatter queue of ls0 is full.                                          |   |   |
| 146          | Tpc_tec2_ls0: sdq full                   | Local store data queue of ls0 is full.                                              |   |   |
| 147          | Tpc_tec2_ls0: load valid                 | Count load executed in ls0.                                                         |   |   |
| 148          | Tpc_tec2_ls0: store valid                | Count store executed in ls0.                                                        |   |   |
| 149          | Tpc_tec2_ls1: lsq full                   | Local load/store queue of ls1 is full.                                              |   |   |
| 150          | Tpc_tec2_ls1: load valid                 | Count load executed in ls1.                                                         |   |   |
| 151          | Tpc_tec2_ls1: store valid                | Count store executed in ls1.                                                        |   |   |
| 152          | Tpc_tec3_ls0: lsq full                   | Local load/store queue of ls0 is full.                                              |   |   |
| 153          | Tpc_tec3_ls0: gsq full                   | Local gather/scatter queue of ls0 is full.                                          |   |   |
| 154          | Tpc_tec3_ls0: sdq full                   | Local store data queue of ls0 is full.                                              |   |   |
| 155          | Tpc_tec3_ls0: load valid                 | Count load executed in ls0.                                                         |   |   |
| 156          | Tpc_tec3_ls0: store valid                | Count store executed in ls0.                                                        |   |   |
| 157          | Tpc_tec3_ls1: lsq full                   | Local load/store queue of ls1 is full.                                              |   |   |
| 158          | Tpc_tec3_ls1: load valid                 | Count load executed in ls1.                                                         |   |   |
| 159          | Tpc_tec3_ls1: store valid                | Count store executed in ls1.                                                        |   |   |
| -            | -                                        | -                                                                                   |   |   |

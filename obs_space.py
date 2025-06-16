    def get_next_action(self):
        self.counter += 1
        self.parameters['step_criteria'].succeed()
        self.parameters['step_criteria'] = self.env.event()
        Transport.agents_waiting_for_action.append(self)
        yield self.parameters['continue_criteria']  # 等待环境计算出动作，需要知道小车运输的order和零件的destination
        # action_id = self.next_action[0]  # 环境给出的动作！！！！！！！！！！！！
        if self.agent_type != "TRPO":
            result_order, result_destination = self.agent.act(Transport.all_transp_orders)
            if result_order and result_destination:
                self.next_action_order = result_order
                result_origin = self.next_action_origin = result_order.current_location
                self.next_action_destination = result_destination
                result_valid = self.next_action_valid = True
                self.next_action[0] = result_destination.id
                step_progress_reward = 30 * (
                            self.next_action_order.actual_step / (len(self.next_action_order.prod_steps) - 1))
                # print("step_progress_reward=============================:", step_progress_reward)
                # # 零件等待时间奖励
                # order_waiting_time_reward = max(0.50, 1 / (1 + self.next_action_order.get_total_waiting_time()))
                # print("order_waiting_time=============================:", order_waiting_time_reward)
                self.next_action_reward = step_progress_reward 
            else:
                result_order = None
                result_origin = None
                result_destination = None
                self.next_action_valid = False
                self.next_action[0] = None
                print("无合法动作")
            # for o in self.all_transp_orders:
            #     print("waiting for transp:", o.id,[x.id for x in o.prod_steps],o.current_location.id,o.current_location.type,o.actual_step)
            return result_order, result_destination
        # 若是强化学习代理方法
        result_order = None
        result_origin = None
        result_dest = None
        result_valid=False
        # min_buffer_fill = 3 # 工作站缓冲区容量
        self.latest_reward = 0.0
        # 处理action_id为0的情况，从source缓冲区获取订单
        if self.mapping[self.next_action[0]] == 0:
            source = self.resources['sources'][0]
            if source.buffer_out:
                for order in source.buffer_out:
                    if order.get_next_step().is_free():
                        result_dest = order.get_next_step()
                        result_order = order
                        result_valid = True
                        break  # 找到第一个符合条件的订单就退出循环
        else:
            machine_id = self.mapping[self.next_action[0]]
            # 根据machine_id获取对应的工作站
            machine = next((m for m in chain(self.resources['machines'], self.resources.get(f'machine_group_{machine_id}', [])) 
                        if m.id == machine_id), None)
            # 如果找到工作站且其输出缓冲区有订单
            # 如果一开始只判断工作站上，那么source上永远不会有来运输的
            if machine.buffer_out:
                # 遍历缓冲区中的所有订单
                for order in machine.buffer_out:
                    # 如果下一个处理步骤是sink
                    if order.get_next_step().type == 'sink'and not order.reserved:
                        result_dest = order.get_next_step()
                        result_order = order
                        result_valid = True
                        result_order.reserved = True
                        break
                    # 如果订单的下一个处理步骤的工作站空闲
                    elif order.get_next_step().is_free() and order.reserved==False:
                        result_dest = order.get_next_step()
                        result_order = order
                        result_order.reserved = True
                        result_valid = True
                        break  # 找到第一个符合条件的订单就退出循环
                    # 检查同组是否有空闲人工站
                    elif order.get_next_step().type == "machine" and order.get_next_step().is_free_hum()and not order.reserved:
                        for mach in [x for x in chain(self.resources['machines'],
                                                    self.resources.get(f'machine_group_{order.get_next_step().machine_group}',[]))
                                    if x.machine_group == order.get_next_step().machine_group and x.machine_type == 2]:
                            # 由于只有一台人工工作站,直接将其作为目标工作站
                            result_dest = mach
                            result_order = order
                            result_valid = True
                            result_order.reserved = True
                        if result_order and result_dest: # 一次循环中 
                            break
                    if result_order and result_dest: # 如果找到订单，则跳出外层循环
                        break
        if result_order and result_dest: # 有效动作
            self.next_action_destination = result_dest
            self.next_action_order = result_order
            self.next_action_origin = result_order.current_location
            self.next_action_valid = result_valid
            result_order.prod_steps[result_order.actual_step] = result_dest # 更改产品的路径
            step_progress_reward = 30 * (
                        self.next_action_order.actual_step / (len(self.next_action_order.prod_steps) - 1))
            # print("step_progress_reward=============================:", step_progress_reward)
            # # 零件等待时间奖励
            # order_waiting_time_reward = max(0.50, 1 / (1 + self.next_action_order.get_total_waiting_time()))
            # print("order_waiting_time=============================:", order_waiting_time_reward)
            self.next_action_reward = step_progress_reward 
            # 将订单从transport_orders中pop出去
            Transport.all_transp_orders.pop(Transport.all_transp_orders.index(result_order))
            # result_order.current_location.buffer_out.pop(result_order.current_location.buffer_out.index(result_order))
        else:
            print("invalid action")
            self.next_action_destination = None
            self.next_action_order = None
            self.next_action_origin = None
            self.next_action_valid = False
        return result_order, result_dest #放回零件和destination 
def calculate_state(self):
        """
        状态空间为4*N的一维向量形式
        其中N为工作站的数量，每4个元素代表一个工作站的信息：
        1. 当前工作站的输出缓冲区订单是否可以分配处理
        2. 当前工作站的订单等待时间
        3. 当前工作站的订单剩余处理时间
        4. 当前工作站的订单完成度信息
        """
        # 可将动态增加工作站改为动态
        all_machines = []
        # 添加主要工作站
        all_machines.extend(self.resources['machines'])
        # 按工作站ID排序以确保状态向量的一致性
        all_machines.sort(key=lambda x: x.id)
        num_machines = len(all_machines)
        # 初始化4*(N+1)的一维状态向量--逐渐增加变量 
        state_vector = [0.0] * (4 * (num_machines + 1))
        # state_vector = [0.0] * (num_machines + 1)
        # 处理source缓冲区中的订单
        source_orders = self.resources['sources'][0].buffer_out
        if source_orders:
            # assignable_ratio = 0.0
            # for order in source_orders:
            #     if order.get_next_step().is_free():
            #         assignable_ratio = 1.0
            #         break
            #     elif order.get_next_step().type == "machine" and order.get_next_step().is_free_hum():
            #         assignable_ratio = 1.0
            #         break
            state_vector[0] = assignable_ratio
            # 计算所有订单的平均等待时间
            total_waiting_time = 0.0
            for order in source_orders:
                total_waiting_time += order.get_total_waiting_time()
            avg_waiting_time = total_waiting_time / len(source_orders) if source_orders else 0.0
            normalized_waiting_time = min(avg_waiting_time / 300.0, 1.0)
            state_vector[num_machines + 1] = normalized_waiting_time
            
            # 计算source输出缓冲区的负载情况
            buffer_load = len(source_orders) / 3  # 假设最大容量为10，归一化到[0,1]区间
            state_vector[2 * (num_machines + 1)] = min(buffer_load, 1.0)  # 确保不超过1.0

            # 计算source平均完成度一定为零
            total_completion = 0.0
            state_vector[3 * (num_machines + 1)] = total_completion
            # # 3. 剩余处理时间
            # normalized_time = 1
            # state_vector[2 * (num_machines + 1)] = normalized_time
            # # 4. 完成度信息
            # total_steps = len(order.prod_steps)
            # completed_steps = order.actual_step
            # completion_ratio = completed_steps / total_steps if total_steps > 0 else 0.0
            # state_vector[3 * (num_machines + 1)] = completion_ratio
        # 为每个工作站计算状态信息
        for machine_idx, machine in enumerate(all_machines):
            # 获取该工作站的输出缓冲区订单
            machine_orders = machine.buffer_out
            if machine_orders:
                # 初始化可分配比率
                assignable_ratio = 0.0
                # 遍历所有订单
                for order in machine_orders:
                    # 如果下一个处理步骤是sink
                    if order.get_next_step().type == "sink":
                        assignable_ratio = 1.0
                        break
                    # 判断下一步工作站是否空闲
                    elif order.get_next_step().is_free():
                        assignable_ratio = 1.0
                        break  # 只要有一个订单可分配就跳出循环
                    elif order.get_next_step().type == "machine" and order.get_next_step().is_free_hum():
                        assignable_ratio = 1.0
                        break
                state_vector[machine_idx + 1] = assignable_ratio
                    # 计算工作站中所有订单的平均等待时间
                total_waiting_time = 0.0
                for order in machine_orders:
                    total_waiting_time += order.get_total_waiting_time()
                avg_waiting_time = total_waiting_time / len(machine_orders) if machine_orders else 0.0
                normalized_waiting_time = min(avg_waiting_time / 300.0, 1.0)
                state_vector[machine_idx + num_machines + 2] = normalized_waiting_time
                # 计算工作站输出缓冲区的负载情况
                buffer_load = len(machine_orders) / 4  # 假设最大容量为5，归一化到[0,1]区间
                state_vector[machine_idx + 2 * (num_machines + 1) + 1] = min(buffer_load, 1.0)  # 确保不超过1.0
                # 计算工作站输出缓冲区中所有订单的平均完成度
                total_steps_all_orders = 0
                completed_steps_all_orders = 0
                for order in machine_orders:
                    # 累加所有订单的总步骤数
                    total_steps_all_orders += len(order.prod_steps)
                    # 累加所有订单的已完成步骤数
                    completed_steps_all_orders += order.actual_step
                # 计算整体完成度
                completion_ratio = completed_steps_all_orders / total_steps_all_orders if total_steps_all_orders > 0 else 0.0
                state_vector[machine_idx + 3 * (num_machines + 1) + 1] = completion_ratio
                # # 3. 剩余处理时间：计算该工作站订单的平均剩余处理时间（归一化）
                # remaining_time = self.time_calc.remaining_steps_processing_time(order)
                # avg_waiting_time = self.time_calc.average_remaining_waiting_time()
                # normalized_time = min(remaining_time / avg_waiting_time, 1.0) if avg_waiting_time > 0 else 0.0
                # state_vector[machine_idx + 2 * (num_machines + 1) + 1] = normalized_time
                # # 4. 完成度信息：由于每个工作站的输出缓冲区只有一个订单，直接计算该订单的完成度
                # total_steps = len(order.prod_steps)
                # completed_steps = order.actual_step
                # completion_ratio = completed_steps / total_steps if total_steps > 0 else 0.0
                # state_vector[machine_idx + 3 * (num_machines + 1) + 1] = completion_ratio

        return state_vector

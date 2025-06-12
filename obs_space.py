    def calculate_state(self):
        """
        状态空间为4*N的二维矩阵形式
        其中N为工作站的数量，每一维包括：
        1. 当前工作站的输出缓冲区订单是否可以分配处理
        2. 当前工作站的订单等待时间
        3. 当前工作站的订单剩余处理时间
        4. 当前工作站的订单完成度信息
        """
        # 可将动态增加工作站改为动态
        all_machines = []
        # 添加主要工作站
        all_machines.extend(self.resources['machines'])
        # # 添加动态添加的工作站
        # for group_key in [1, 2, 4, 5, 6]:
        #     dynamic_machines = self.resources.get(f'machine_group_{group_key}', [])
        #     all_machines.extend(dynamic_machines)
        # 按工作站ID排序以确保状态向量的一致性
        all_machines.sort(key=lambda x: x.id)
        num_machines = len(all_machines)
        # 初始化4*N的状态矩阵
        state_matrix = [[0.0] * 4 for _ in range(num_machines+1)]
        # 处理source缓冲区中的订单
        source_orders = self.resources['sources'][0].buffer_out
        if source_orders:
            order = source_orders[0]
            # 1. 可分配处理状态
            if order.get_next_step().is_free():
                assignable_ratio = 1.0
            elif order.get_next_step().type == "machine"and order.get_next_step().is_free_hum():
                assignable_ratio = 1.0
            else:
                assignable_ratio = 0.0
            state_matrix[0][0] = assignable_ratio
            # 2. 等待时间
            waiting_time = order.get_total_waiting_time()
            normalized_waiting_time = min(waiting_time / 300.0, 1.0)
            state_matrix[0][1] = normalized_waiting_time
            
            # 3. 剩余处理时间
            # remaining_time = self.time_calc.remaining_steps_processing_time(order)
            # avg_waiting_time = self.time_calc.average_remaining_waiting_time()
            normalized_time = 1
            state_matrix[0][2] = normalized_time
            # 4. 完成度信息
            total_steps = len(order.prod_steps)
            completed_steps = order.actual_step
            completion_ratio = completed_steps / total_steps if total_steps > 0 else 0.0
            state_matrix[0][3] = completion_ratio
        else:
            state_matrix[0] = [0.0, 0.0, 0.0, 0.0]
        # 为每个工作站计算状态信息
        for machine_idx, machine in enumerate(all_machines):
            # 获取该工作站的输出缓冲区订单
            machine_orders = machine.buffer_out
            if machine_orders:
                # 1. 可分配处理状态：假设输出缓冲区最多只有一个订单
                order = machine_orders[0]
                # 判断下一步工作站是否空闲
                if order.get_next_step().is_free() or order.get_next_step().type == "sink":
                    assignable_ratio = 1.0
                elif order.get_next_step().type == "machine" and order.get_next_step().is_free_hum():
                    assignable_ratio = 1.0
                else:
                    assignable_ratio = 0.0
                state_matrix[machine_idx + 1][0] = assignable_ratio
                # 2. 等待时间：直接取该订单的等待时间（归一化），等待时间不能超过MAX_WAITING_TIME
                waiting_time =order.get_total_waiting_time()
                normalized_waiting_time = min(waiting_time / 300.0, 1.0)
                state_matrix[machine_idx + 1][1] = normalized_waiting_time
                # 3. 剩余处理时间：计算该工作站订单的平均剩余处理时间（归一化）
                remaining_time = self.time_calc.remaining_steps_processing_time(order)
                avg_waiting_time = self.time_calc.average_remaining_waiting_time()
                normalized_time = min(remaining_time / avg_waiting_time, 1.0) if avg_waiting_time > 0 else 0.0
                state_matrix[machine_idx + 1][2] = normalized_time
                # 4. 完成度信息：由于每个工作站的输出缓冲区只有一个订单，直接计算该订单的完成度
                total_steps = len(order.prod_steps)
                completed_steps = order.actual_step
                completion_ratio = completed_steps / total_steps if total_steps > 0 else 0.0
                state_matrix[machine_idx + 1][3] = completion_ratio
            else:
                state_matrix[machine_idx + 1] = [0.0, 0.0, 0.0, 0.0]
        # 直接返回二维状态矩阵
        result_state = state_matrix
        return result_state

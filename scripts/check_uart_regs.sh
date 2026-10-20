#!/bin/bash
# 检查 r_uart0 的寄存器状态，判断 UART 是否真正可用

BASE=0xc0881000

echo "=== 设置波特率前 LCR ==="
busybox devmem 0xc088100c

echo "=== 执行 stty 115200 raw ==="
stty -F /dev/ttyS10 115200 raw -echo
echo "exit=$?"

echo "=== 设置波特率后 LCR ==="
busybox devmem 0xc088100c

echo "=== 全部寄存器 (offset -> value) ==="
for off in 0x00 0x04 0x08 0x0c 0x10 0x14 0x18 0x1c; do
    addr=$((BASE + off))
    hexaddr=$(printf "0x%x" $addr)
    val=$(busybox devmem $hexaddr)
    echo "  $hexaddr (off $off): $val"
done

echo "=== 写入测试：往 THR 写 0x55，再读回 LSR ==="
busybox devmem 0xc0881000 32 0x55
busybox devmem 0xc0881014

echo "=== 写 MCR 测试 ==="
busybox devmem 0xc0881010 32 0x03
busybox devmem 0xc0881010

echo "=== rtpm 状态 ==="
cat /sys/devices/platform/soc/c0881000.r_uart0/power/runtime_status 2>/dev/null || echo "no runtime_status"

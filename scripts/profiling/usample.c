// Minimal user-space sampling profiler for machines without `perf`.
//
// It samples the instruction pointer of one child process with perf_event_open
// in user-only mode, which works unprivileged at kernel.perf_event_paranoid=2
// (where samply and `perf record` are refused). scripts/profile_sample.py
// builds this, runs it, and symbolizes the output.
//
// usage: usample <out.txt> <freq_hz> -- cmd args...
// Output: one line per sample, "ip caller1 caller2 ..." in hex, plus
// "M start len pgoff path" lines for each executable mapping.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

static long perf_open(struct perf_event_attr *a, pid_t pid, int cpu, int grp, unsigned long fl) {
    return syscall(SYS_perf_event_open, a, pid, cpu, grp, fl);
}

#define PAGES 128 /* data pages, power of two; small enough for the default perf mlock budget */

static FILE *out;
static uint64_t nsamples = 0, nlost = 0;

static void drain(struct perf_event_mmap_page *mp, size_t page) {
    char *data = (char *)mp + page;
    uint64_t size = (uint64_t)PAGES * page;
    uint64_t head = __atomic_load_n(&mp->data_head, __ATOMIC_ACQUIRE);
    uint64_t tail = mp->data_tail;
    static char buf[1 << 16];
    while (tail < head) {
        struct perf_event_header hdr;
        for (size_t i = 0; i < sizeof hdr; i++) ((char *)&hdr)[i] = data[(tail + i) % size];
        if (hdr.size == 0 || hdr.size > sizeof buf) break;
        for (size_t i = 0; i < hdr.size; i++) buf[i] = data[(tail + i) % size];
        char *p = buf + sizeof hdr;
        if (hdr.type == PERF_RECORD_SAMPLE) {
            uint64_t ip = *(uint64_t *)p; p += 8;
            uint64_t nr = *(uint64_t *)p; p += 8;
            fprintf(out, "%lx", (unsigned long)ip);
            for (uint64_t i = 0; i < nr && i < 64; i++) {
                uint64_t c = ((uint64_t *)p)[i];
                if (c >= (uint64_t)-4096) continue; /* context markers */
                if (i == 0 && c == ip) continue;
                fprintf(out, " %lx", (unsigned long)c);
            }
            fputc('\n', out);
            nsamples++;
        } else if (hdr.type == PERF_RECORD_MMAP) {
            // pid, tid, addr, len, pgoff, filename
            uint64_t addr = *(uint64_t *)(p + 8), len = *(uint64_t *)(p + 16), pgoff = *(uint64_t *)(p + 24);
            fprintf(out, "M %lx %lx %lx %s\n", (unsigned long)addr, (unsigned long)len, (unsigned long)pgoff, p + 32);
        } else if (hdr.type == PERF_RECORD_LOST) {
            nlost += *(uint64_t *)(p + 8);
        }
        tail += hdr.size;
    }
    __atomic_store_n(&mp->data_tail, tail, __ATOMIC_RELEASE);
}

int main(int argc, char **argv) {
    if (argc < 5 || strcmp(argv[3], "--") != 0) {
        fprintf(stderr, "usage: %s out.txt freq_hz -- cmd...\n", argv[0]);
        return 2;
    }
    out = fopen(argv[1], "w");
    if (!out) { perror("fopen"); return 2; }
    int freq = atoi(argv[2]);
    int pfd[2];
    if (pipe(pfd)) { perror("pipe"); return 2; }
    pid_t child = fork();
    if (child == 0) {
        close(pfd[1]);
        char c;
        if (read(pfd[0], &c, 1) != 1) _exit(127);
        execvp(argv[4], argv + 4);
        perror("execvp");
        _exit(127);
    }
    close(pfd[0]);
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = PERF_TYPE_SOFTWARE;
    a.config = PERF_COUNT_SW_CPU_CLOCK;
    a.freq = 1;
    a.sample_freq = freq;
    a.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_CALLCHAIN;
    a.disabled = 1;
    a.enable_on_exec = 1;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.exclude_callchain_kernel = 1;
    a.mmap = 1;
    a.inherit = 0;
    int fd = perf_open(&a, child, -1, -1, 0);
    if (fd < 0) { perror("perf_event_open"); kill(child, SIGKILL); return 2; }
    size_t page = sysconf(_SC_PAGESIZE);
    struct perf_event_mmap_page *mp = mmap(NULL, (PAGES + 1) * page, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mp == MAP_FAILED) { perror("mmap"); kill(child, SIGKILL); return 2; }
    if (write(pfd[1], "x", 1) != 1) return 2;
    int status = 0;
    for (;;) {
        struct pollfd pf = {fd, POLLIN, 0};
        poll(&pf, 1, 20);
        drain(mp, page);
        pid_t r = waitpid(child, &status, WNOHANG);
        if (r == child) break;
    }
    drain(mp, page);
    fclose(out);
    fprintf(stderr, "usample: %lu samples, %lu lost\n", (unsigned long)nsamples, (unsigned long)nlost);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

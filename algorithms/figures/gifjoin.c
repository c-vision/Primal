/*
 * PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.  A copy of the License
 * is in the repository root (LICENSE) and at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

/* gifjoin.c - joins single-image GIF files into one looping animated GIF.
 *
 *   gifjoin out.gif CS:frame1.gif CS:frame2.gif ...
 *
 * CS is the display time of the frame in hundredths of a second.  The frames must have
 * the same size.  The compressed image data of every frame is copied unchanged; only the
 * container is rewritten (each frame keeps its own colour table as a local table), so
 * no LZW code is needed here.  Used by make_gifs.sh to assemble the animations of this
 * directory from frames rendered by make_animations.c.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gifjoin: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "gifjoin: cannot read %s\n", path); exit(1); }
    fclose(f);
    *len = (size_t)n;
    return b;
}

static void die(const char *path, const char *why) { fprintf(stderr, "gifjoin: %s: %s\n", path, why); exit(1); }

int main(int argc, char **argv) {
    if (argc < 3) { fputs("usage: gifjoin out.gif CS:frame.gif ...\n", stderr); return 2; }
    FILE *out = fopen(argv[1], "wb");
    if (!out) { fprintf(stderr, "gifjoin: cannot write %s\n", argv[1]); return 1; }
    int W = 0, H = 0;
    for (int a = 2; a < argc; a++) {
        char *colon = strchr(argv[a], ':');
        if (!colon) die(argv[a], "expected CS:file");
        int cs = atoi(argv[a]);
        const char *path = colon + 1;
        size_t n; unsigned char *b = slurp(path, &n);
        if (n < 14 || memcmp(b, "GIF8", 4) != 0) die(path, "not a GIF");
        int w = b[6] | b[7] << 8, h = b[8] | b[9] << 8;
        unsigned flags = b[10];
        size_t pos = 13;
        unsigned char *gct = NULL; size_t gctlen = 0;
        if (flags & 0x80) { gctlen = 3u << ((flags & 7) + 1); gct = b + pos; pos += gctlen; }
        if (a == 2) {
            W = w; H = h;
            unsigned char hd[] = { 'G','I','F','8','9','a', (unsigned char)(W & 255), (unsigned char)(W >> 8),
                                   (unsigned char)(H & 255), (unsigned char)(H >> 8), 0x70, 0, 0 };
            fwrite(hd, 1, sizeof hd, out);
            static const unsigned char loop[] = { 0x21, 0xFF, 0x0B, 'N','E','T','S','C','A','P','E','2','.','0',
                                                  0x03, 0x01, 0x00, 0x00, 0x00 };
            fwrite(loop, 1, sizeof loop, out);
        } else if (w != W || h != H) die(path, "frame size differs from the first frame");
        int done = 0;
        while (pos < n && !done) {
            unsigned char t = b[pos++];
            if (t == 0x21) {                           /* extension: skip */
                pos++;
                while (pos < n && b[pos]) pos += (size_t)b[pos] + 1;
                pos++;
            } else if (t == 0x2C) {                    /* image */
                if (pos + 9 > n) die(path, "truncated");
                unsigned char id[9]; memcpy(id, b + pos, 9); pos += 9;
                unsigned char *ct = gct; size_t ctlen = gctlen; unsigned cflags = id[8];
                if (cflags & 0x80) { ctlen = 3u << ((cflags & 7) + 1); ct = b + pos; pos += ctlen; }
                if (!ct) die(path, "no colour table");
                unsigned bits = 0; while ((2u << bits) < ctlen / 3) bits++;
                id[8] = (unsigned char)((cflags & 0x40) | 0x80 | bits);      /* keep interlace, force local table */
                unsigned char gce[] = { 0x21, 0xF9, 0x04, 0x04, (unsigned char)(cs & 255), (unsigned char)(cs >> 8), 0, 0 };
                fwrite(gce, 1, sizeof gce, out);
                fputc(0x2C, out); fwrite(id, 1, 9, out);
                fwrite(ct, 1, ctlen, out);
                size_t start = pos; pos++;                                   /* LZW minimum code size */
                while (pos < n && b[pos]) pos += (size_t)b[pos] + 1;
                pos++;
                if (pos > n) die(path, "truncated image data");
                fwrite(b + start, 1, pos - start, out);
                done = 1;
            } else if (t == 0x3B) break;
            else die(path, "unexpected block");
        }
        if (!done) die(path, "no image");
        free(b);
    }
    fputc(0x3B, out);
    fclose(out);
    return 0;
}


#pragma once
static void reorder_b(int8_t* b, int8_t* sb, const int k, const int n, const int ldx) {

    int i = 0;
    for (; i+3 < n; i += 4) {
        int8_t *p0 = b + i;
        int8_t *p1 = b + 1 * ldx + i;
        int8_t *p2 = b + 2 * ldx + i;
        int8_t *p3 = b + 3 * ldx + i;

        int8_t *p4 = b + 4 * ldx + i;
        int8_t *p5 = b + 5 * ldx + i;
        int8_t *p6 = b + 6 * ldx + i;
        int8_t *p7 = b + 7 * ldx + i;

        int j = 0;
        for (; j+7 < k; j += 8) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];
            sb[4]  = p4[0];
            sb[5]  = p5[0];
            sb[6]  = p6[0];
            sb[7]  = p7[0];

            sb[8]  = p0[1];
            sb[9]  = p1[1];
            sb[10] = p2[1];
            sb[11] = p3[1];
            sb[12] = p4[1];
            sb[13] = p5[1];
            sb[14] = p6[1];
            sb[15] = p7[1];

            sb[16] = p0[2];
            sb[17] = p1[2];
            sb[18] = p2[2];
            sb[19] = p3[2];
            sb[20] = p4[2];
            sb[21] = p5[2];
            sb[22] = p6[2];
            sb[23] = p7[2];

            sb[24] = p0[3];
            sb[25] = p1[3];
            sb[26] = p2[3];
            sb[27] = p3[3];
            sb[28] = p4[3];
            sb[29] = p5[3];
            sb[30] = p6[3];
            sb[31] = p7[3];

            sb += 32;
            p0 += 8 * ldx;
            p1 += 8 * ldx;
            p2 += 8 * ldx;
            p3 += 8 * ldx;
            p4 += 8 * ldx;
            p5 += 8 * ldx;
            p6 += 8 * ldx;
            p7 += 8 * ldx;
        }
        for (; j+3 < k; j += 4) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];

            sb[4]  = p0[1];
            sb[5]  = p1[1];
            sb[6]  = p2[1];
            sb[7]  = p3[1];

            sb[8]  = p0[2];
            sb[9]  = p1[2];
            sb[10] = p2[2];
            sb[11] = p3[2];

            sb[12] = p0[3];
            sb[13] = p1[3];
            sb[14] = p2[3];
            sb[15] = p3[3];

            sb += 16;
            p0 += 4 * ldx;
            p1 += 4 * ldx;
            p2 += 4 * ldx;
            p3 += 4 * ldx;
        }
        for (; j+1 < k; j += 2) {
            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p0[1];
            sb[3] = p1[1];
            sb[4] = p0[2];
            sb[5] = p1[2];
            sb[6] = p0[3];
            sb[7] = p1[3];

            sb += 8;
            p0 += 2 * ldx;
            p1 += 2 * ldx;
        }
        for (; j < k; ++j) {
            sb[0] = p0[0];
            sb[1] = p0[1];
            sb[2] = p0[2];
            sb[3] = p0[3];

            sb += 4;
            p0 += ldx;
        }
    }
    for (; i+1 < n; i += 2) {
        int8_t *p0 = b + i;
        int8_t *p1 = b + 1 * ldx + i;
        int8_t *p2 = b + 2 * ldx + i;
        int8_t *p3 = b + 3 * ldx + i;

        int8_t *p4 = b + 4 * ldx + i;
        int8_t *p5 = b + 5 * ldx + i;
        int8_t *p6 = b + 6 * ldx + i;
        int8_t *p7 = b + 7 * ldx + i;

        int j = 0;
        for (; j+7 < k; j += 8) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];
            sb[4]  = p4[0];
            sb[5]  = p5[0];
            sb[6]  = p6[0];
            sb[7]  = p7[0];

            sb[8]  = p0[1];
            sb[9]  = p1[1];
            sb[10] = p2[1];
            sb[11] = p3[1];
            sb[12] = p4[1];
            sb[13] = p5[1];
            sb[14] = p6[1];
            sb[15] = p7[1];

            sb += 16;
            p0 += 8 * ldx;
            p1 += 8 * ldx;
            p2 += 8 * ldx;
            p3 += 8 * ldx;
            p4 += 8 * ldx;
            p5 += 8 * ldx;
            p6 += 8 * ldx;
            p7 += 8 * ldx;
        }
        for (; j+3 < k; j += 4) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];

            sb[4]  = p0[1];
            sb[5]  = p1[1];
            sb[6]  = p2[1];
            sb[7]  = p3[1];

            sb += 8;
            p0 += 4 * ldx;
            p1 += 4 * ldx;
            p2 += 4 * ldx;
            p3 += 4 * ldx;
        }
        for (; j+1 < k; j += 2) {
            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p0[1];
            sb[3] = p1[1];

            sb += 4;
            p0 += 2 * ldx;
            p1 += 2 * ldx;
        }
        for (; j < k; ++j) {
            sb[0] = p0[0];
            sb[1] = p0[1];

            sb += 2;
            p0 += ldx;
        }
    }
    for (; i < n; ++i) {
        int8_t *p0 = b + i;
        int8_t *p1 = b + 1 * ldx + i;
        int8_t *p2 = b + 2 * ldx + i;
        int8_t *p3 = b + 3 * ldx + i;
        int8_t *p4 = b + 4 * ldx + i;
        int8_t *p5 = b + 5 * ldx + i;
        int8_t *p6 = b + 6 * ldx + i;
        int8_t *p7 = b + 7 * ldx + i;

        int j = 0;
        for (; j+7 < k; j += 8) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];
            sb[4]  = p4[0];
            sb[5]  = p5[0];
            sb[6]  = p6[0];
            sb[7]  = p7[0];

            sb += 8;
            p0 += 8 * ldx;
            p1 += 8 * ldx;
            p2 += 8 * ldx;
            p3 += 8 * ldx;
            p4 += 8 * ldx;
            p5 += 8 * ldx;
            p6 += 8 * ldx;
            p7 += 8 * ldx;
        }
        for (; j+3 < k; j += 4) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];

            sb += 4;
            p0 += 4 * ldx;
            p1 += 4 * ldx;
            p2 += 4 * ldx;
            p3 += 4 * ldx;
        }
        for (; j+1 < k; j += 2) {
            sb[0] = p0[0];
            sb[1] = p1[0];

            sb += 2;
            p0 += 2 * ldx;
            p1 += 2 * ldx;
        }
        for (; j < k; ++j) {
            sb[0] = p0[0];

            sb += 1;
            p0 += ldx;
        }
    }
}

// Common scene definitions
enum ShapeType {
    Circle,
    Polyline,
    QuadraticBezier
};

struct Shape {
    ShapeType type;
    // Index the shape into the type-specific arrays
    // (circles, polylines, or quadratic_beziers)
    int index;

    bool use_fill;
    float4 fill_rgba;
    bool use_stroke;
    float4 stroke_rgba;
    float stroke_width;
    // Maps screen/world-space positions to the shape's object space.
    float3x3 world_to_obj;
};

struct Circle {
    float2 center;
    float radius;
};

struct Polyline {
    int pts_begin;
    int pts_end;
    bool closed;
};

struct QuadraticBezier {
    int pts_begin;
    int pts_end;
    bool closed;
};

static const float PI = 3.14159265358979323846;

// Polynomial solvers
// Return true if a root is found.
// The two real roots are stored in t
bool solve_quadratic(float a, float b, float c,
                     out float2 t) {
    // From https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/math.h#L613
    if (a == 0) {
        if (b == 0) {
            t = float2(0.0, 0.0);
            return false;
        }
        t = float2(-c / b, -c / b);
        return true;
    }

    float discrim = b * b - 4 * a * c;
    if (discrim < 0) {
        t = float2(0, 0);
        return false;
    }
    float root_discrim = sqrt(discrim);

    float q;
    if (b < 0) {
        q = -0.5f * (b - root_discrim);
    } else {
        q = -0.5f * (b + root_discrim);
    }
    t.x = q / a;
    t.y = c / q;
    return true;
}

// Returns real candidate roots needed for finding extrema.
// Repeated roots may be omitted.
int solve_cubic(float a, float b, float c, float d, out float3 t) {
    if (abs(a) < 1e-6) {
        if (solve_quadratic(b, c, d, t.xy)) {
            return 2;
        } else {
            return 0;
        }
    }
    // normalize cubic equation
    b /= a;
    c /= a;
    d /= a;
    float Q = (b * b - 3 * c) / 9.0;
    float R = (2 * b * b * b - 9 * b * c + 27 * d) / 54.0;
    if (R * R < Q * Q * Q) {
        // 3 real roots
        float theta = acos(R / sqrt(Q * Q * Q));
        t.x = -2.0 * sqrt(Q) * cos(theta / 3.0) - b / 3.0;
        t.y = -2.0 * sqrt(Q) * cos((theta - 2.0 * PI) / 3.0) - b / 3.0;
        t.z = -2.0 * sqrt(Q) * cos((theta + 2.0 * PI) / 3.0) - b / 3.0;
        return 3;
    } else {
        float A = R > 0 ? -pow(R + sqrt(R * R - Q * Q * Q), 1.0/3.0):
                           pow(-R + sqrt(R * R - Q * Q * Q), 1.0/3.0);
        float B = abs(A) > 1e-6 ? Q / A : 0.0;
        t.x = (A + B) - b / 3.0;
        return 1;
    }
}

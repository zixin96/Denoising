#include "denoiser.h"

Denoiser::Denoiser() : m_useTemportal(false) {}

// frameInfo: frame t  
// m_accColor: results from t - 1
void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
#pragma omp parallel for
    // for each pixel i, represented by (x, y)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // forward reprojection: works from frame t - 1 and projects them into frame t
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.0f);

            Float3 cur_pos_i = frameInfo.m_position(x, y);
            float cur_id_i = frameInfo.m_id(x, y);
            if (cur_id_i < 0.0f) {
                // background has id -1, skip it
                continue;
            }

            Matrix4x4 curWorldToObj = Inverse(frameInfo.m_matrix[(int)cur_id_i]);
            Matrix4x4 preObjToWorld = m_preFrameInfo.m_matrix[(int)cur_id_i];

            // 1. find the position of pixel i in previous frame
            Float3 pre_coord_i = preWorldToScreen(
                preObjToWorld(curWorldToObj(cur_pos_i, Float3::EType::Point),
                              Float3::EType::Point),
                Float3::EType::Point);

            bool withinScreen = (int)pre_coord_i.x >= 0.0f &&
                                (int)pre_coord_i.x < width &&
                                (int)pre_coord_i.y >= 0.0f && (int)pre_coord_i.y < height;
            if (withinScreen) {
                bool sameID = cur_id_i ==
                              m_preFrameInfo.m_id((int)pre_coord_i.x, (int)pre_coord_i.y);
                // 2. check validity
                m_valid(x, y) = withinScreen && sameID;

                // 3. project results to the current frame
                m_misc(x, y) = m_accColor((int)pre_coord_i.x, (int)pre_coord_i.y);
            }
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 3;
    // NxN filter
    int N = 2 * kernelRadius + 1;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Temporal clamp
            Float3 preFilteredColor = m_accColor(x, y);

            // 1. Outlier detection: compute mean and variance of pixel's NxN neighborhood 
            int topLeftX = x - kernelRadius;
            int topLeftY = y - kernelRadius;
            Float3 colorSum = Float3(0.0f);
            float totalPixels = 0.0f;
            for (int innerY = topLeftY; innerY < topLeftY + N; innerY++) {
                for (int innerX = topLeftX; innerX < topLeftX + N; innerX++) {
                    if (innerX >= 0 && innerX < width && innerY >= 0 && innerY < height) {
                        colorSum += m_accColor(innerX, innerY);
                        totalPixels++;
                    }
                }
            }
            Float3 colorMean = colorSum / totalPixels;

            Float3 colorVariance = Float3(0.0f);
            for (int innerY = topLeftY; innerY < topLeftY + N; innerY++) {
                for (int innerX = topLeftX; innerX < topLeftX + N; innerX++) {
                    if (innerX >= 0 && innerX < width && innerY >= 0 && innerY < height) {
                        colorVariance += Sqr((m_accColor(innerX, innerY) - colorMean));
                    }
                }
            }
            colorVariance = colorVariance / totalPixels;
            Float3 colorSd = SafeSqrt(colorVariance);

            // 2. Outlier Clamping
            preFilteredColor = Clamp(preFilteredColor, colorMean - colorSd * m_colorBoxK,
                                     colorMean + colorSd * m_colorBoxK);

            // Exponential moving average
            float alpha = 1.0f;
            if (m_valid(x, y)) {
                alpha = m_alpha;
            }
            m_misc(x, y) = Lerp(preFilteredColor, curFilteredColor(x, y), alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 16;
    // int kernelRadius = 3;
    // NxN filter
    int N = 2 * kernelRadius + 1;

#pragma omp parallel for
    // for each pixel i, represented by (x, y)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // joint bilateral filter
            Float3 coord_i = Float3((float)x, (float)y, 0.0f);
            Float3 color_i = frameInfo.m_beauty(x, y);
            Float3 normal_i = frameInfo.m_normal(x, y);
            float depth_i = frameInfo.m_depth(x, y);
            Float3 pos_i = frameInfo.m_position(x, y);

            float sumOfWeights = 0.0f;
            Float3 sumOfWeightedValues = Float3(0.0f);
            int topLeftX = x - kernelRadius;
            int topLeftY = y - kernelRadius;

            // TODO: for large filters, consider using a-trous wavelet (1:00:40, lecture 13)
            // TODO: for small filters, use separate pass (though in theory, it doesn't with with joint bilateral filter)

            // for each pixel j, represented by (innerX, innerY), around i
            for (int innerY = topLeftY; innerY < topLeftY + N;
                 innerY++) {
                for (int innerX = topLeftX; innerX < topLeftX + N;
                     innerX++) {
            
                    if (innerX >= 0 && innerX < width && innerY >= 0 && innerY <
                    height) {
                        Float3 coord_j = Float3((float)innerX, (float)innerY, 0.0f);
                        Float3 color_j = frameInfo.m_beauty(innerX, innerY);
                        Float3 normal_j = frameInfo.m_normal(innerX, innerY);
                        float depth_j = frameInfo.m_depth(innerX, innerY);
                        Float3 pos_j = frameInfo.m_position(innerX, innerY);
            
                        float coordWeight =
                            -Sqr(Distance(coord_i, coord_j)) / (2 * Sqr(m_sigmaCoord));
                        float colorWeight =
                            -Sqr(Distance(color_i, color_j)) / (2 * Sqr(m_sigmaColor));
                        float normalWeight = -Sqr(SafeAcos(Dot(normal_i, normal_j))) /
                                             (2 * Sqr(m_sigmaNormal));
            
                        float depthWeight = 0.0f;
                        // cannot normalize a vector of size 0
                        if (Length(pos_j - pos_i) != 0.0f) {
                            depthWeight = -Sqr(Dot(normal_i, Normalize(pos_j - pos_i)))
                            /
                                          (2 * Sqr(m_sigmaPlane));
                        }
            
                        // compute the weight
                        float weight =
                            exp(coordWeight + colorWeight + normalWeight +
                            depthWeight);
                        Float3 weightedValue = color_j * weight;
            
                        sumOfWeightedValues += weightedValue;
                        sumOfWeights += weight;
                    }
                }
            }

            if (sumOfWeights == 0.0f) {
                filteredImage(x, y) = Float3(0.0f);
            } else {
                filteredImage(x, y) = sumOfWeightedValues / sumOfWeights;
            }
        }
    }
    return filteredImage;
}


void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrameDenoiseOnly(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Filter(frameInfo);
    m_accColor.Copy(filteredColor);
    return m_accColor;
}


Buffer2D<Float3> Denoiser::ProcessFrameReproAccuOnly(const FrameInfo &frameInfo) {
    Buffer2D<Float3> filteredColor;
    filteredColor.Copy(frameInfo.m_beauty);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Filter(frameInfo);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}

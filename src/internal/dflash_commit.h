#pragma once

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace hunyuan_ocr::internal {

struct DFlashCommitPlan {
    int emitted_drafts = 0;
    int commit_count = 1;
    int next_token = 0;
    int stop_token = -1;
    bool accepted_eos = false;
};

inline DFlashCommitPlan make_dflash_commit_plan(int accepted_drafts, int remaining,
                                                const std::vector<int>& verification_ids,
                                                const std::vector<int>& posterior,
                                                const std::unordered_set<int>& eos_ids) {
    if (accepted_drafts < 0 || remaining < 0 ||
        accepted_drafts >= static_cast<int>(verification_ids.size())) {
        throw std::invalid_argument("invalid DFlash commit bounds");
    }

    DFlashCommitPlan plan;
    plan.emitted_drafts = std::min(accepted_drafts, remaining);
    for (int index = 1; index <= plan.emitted_drafts; ++index) {
        if (eos_ids.count(verification_ids[index])) {
            plan.emitted_drafts = index;
            plan.accepted_eos = true;
            plan.stop_token = verification_ids[index];
            break;
        }
    }
    plan.commit_count = plan.emitted_drafts + 1;
    if (plan.accepted_eos) {
        plan.next_token = plan.stop_token;
    } else {
        if (plan.commit_count > static_cast<int>(posterior.size())) {
            throw std::invalid_argument("DFlash posterior is smaller than commit count");
        }
        plan.next_token = posterior[plan.commit_count - 1];
    }
    return plan;
}

}  // namespace hunyuan_ocr::internal
